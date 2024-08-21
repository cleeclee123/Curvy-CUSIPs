import asyncio
import http
import os
import shutil
from typing import Dict, List, TypeAlias, Optional

import aiohttp
import pandas as pd

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def get_treasurygov_header(year: int, cj: http.cookies = None) -> Dict[str, str]:
    cookie_str = ""
    if cj:
        cookies = {
            cookie.name: cookie.value
            for cookie in cj
            if "home.treasury.gov" in cookie.domain
        }
        cookie_str = "; ".join([f"{key}={value}" for key, value in cookies.items()])

    headers = {
        "authority": "home.treasury.gov",
        "method": "GET",
        "path": f"/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}",
        "scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Cookie": cookie_str,
        "Dnt": "1",
        "Sec-Ch-Ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "Windows",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    }

    if cookie_str == "":
        del headers["Cookie"]

    return headers


def multi_download_year_treasury_par_yield_curve_rate(
    years: List[int],
    raw_path: str,
    download=False,
    real_par_yields=False,
    cj: http.cookies = None,
    run_all=False,
    verbose=False,
    proxy: Optional[str] = None,
) -> pd.DataFrame:
    async def fetch_from_treasurygov(
        session: aiohttp.ClientSession, url: str, curr_year: int
    ) -> pd.DataFrame:
        try:
            headers = get_treasurygov_header(curr_year, cj)
            treasurygov_data_type = "".join(url.split("?type=")[1].split("&field")[0])
            full_file_path = os.path.join(
                raw_path, "temp", f"{treasurygov_data_type}.csv"
            )
            async with session.get(url, headers=headers, proxy=proxy) as response:
                if response.status == 200:
                    with open(full_file_path, "wb") as f:
                        chunk_size = 8192
                        while True:
                            chunk = await response.content.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                    return {
                        treasurygov_data_type: await convert_csv_to_excel(
                            full_file_path
                        )
                    }
                else:
                    raise Exception(f"Bad Status: {response.status}")
        except Exception as e:
            print(e) if verbose else None
            return {treasurygov_data_type: pd.DataFrame()}

    async def convert_csv_to_excel(full_file_path: str | None) -> str:
        if not full_file_path:
            return

        copy = full_file_path
        rdir_path = copy.split("\\")
        rdir_path.remove("temp")
        renamed = str.join("\\", rdir_path)
        renamed = f"{renamed.split('.')[0]}.xlsx"

        df_temp = pd.read_csv(full_file_path)
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])
        df_temp["Date"] = df_temp["Date"].dt.strftime("%Y-%m-%d")
        if download:
            df_temp.to_excel(f"{renamed.split('.')[0]}.xlsx", index=False)
        os.remove(full_file_path)
        return df_temp

    async def get_promises(session: aiohttp.ClientSession):
        tasks = []
        for year in years:
            daily_par_yield_curve_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
            daily_par_real_yield_curve_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value={year}&amp;page&amp;_format=csv"
            daily_treasury_bill_rates_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_bill_rates&field_tdr_date_value={year}&page&_format=csv"
            daily_treaury_long_term_rates_extrapolation_factors_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_long_term_rate&field_tdr_date_value={year}&page&_format=csv"
            daily_treasury_real_long_term_rates_averages = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_long_term&field_tdr_date_value={year}&page&_format=csv"
            if run_all:
                tasks.extend(
                    [
                        fetch_from_treasurygov(
                            session, daily_par_yield_curve_url, year
                        ),
                        fetch_from_treasurygov(
                            session, daily_par_real_yield_curve_url, year
                        ),
                        fetch_from_treasurygov(
                            session, daily_treasury_bill_rates_url, year
                        ),
                        fetch_from_treasurygov(
                            session,
                            daily_treaury_long_term_rates_extrapolation_factors_url,
                            year,
                        ),
                        fetch_from_treasurygov(
                            session, daily_treasury_real_long_term_rates_averages, year
                        ),
                    ]
                )
            else:
                curr_url = (
                    daily_par_yield_curve_url
                    if not real_par_yields
                    else daily_par_real_yield_curve_url
                )
                task = fetch_from_treasurygov(session, curr_url, year)
                tasks.append(task)

        return await asyncio.gather(*tasks)

    async def run_fetch_all() -> List[pd.DataFrame]:
        async with aiohttp.ClientSession() as session:
            all_data = await get_promises(session)
            return all_data

    try:
        os.mkdir(f"{raw_path}/temp")
        dfs: List[Dict[str, pd.DataFrame]] = asyncio.run(run_fetch_all())
        shutil.rmtree(f"{raw_path}/temp")

        if not run_all:
            dfs = [next(iter(dictionary.values())) for dictionary in dfs]
            yield_df = pd.concat(dfs, ignore_index=True)
            return yield_df

        organized_by_ust_type_dict: Dict[str, List[pd.DataFrame]] = {}
        for dictionary in dfs:
            ust_data_type, df = next(iter(dictionary)), next(iter(dictionary.values()))
            if not ust_data_type or df is None or df.empty:
                continue
            if ust_data_type not in organized_by_ust_type_dict:
                organized_by_ust_type_dict[ust_data_type] = []
            organized_by_ust_type_dict[ust_data_type].append(df)

        organized_by_ust_type_df_dict_concated: Dict[str, pd.DataFrame] = {}
        for ust_data_type in organized_by_ust_type_dict.keys():
            dfs = organized_by_ust_type_dict[ust_data_type]
            concated_df = pd.concat(dfs, ignore_index=True)
            organized_by_ust_type_df_dict_concated[ust_data_type] = concated_df

        return organized_by_ust_type_df_dict_concated
    except Exception as e:
        if os.path.exists(f"{raw_path}/temp"):
            shutil.rmtree(f"{raw_path}/temp")
        raise e
