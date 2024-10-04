import pandas as pd
import requests
import asyncio
import http
import aiohttp
import webbrowser
import os
import time
from datetime import date, datetime
from typing import Tuple, List, Dict


def is_downloadable(url):
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get("content-type")
    if "text" in content_type.lower():
        return False
    if "html" in content_type.lower():
        return False
    return True


def get_yahoofinance_download_auth(
    path: str, cj: http.cookiejar = None, run_crumb=False
) -> Tuple[dict, str] | dict:
    cookie_str = ""
    if cj:
        cookies = {
            cookie.name: cookie.value for cookie in cj if "yahoo" in cookie.domain
        }
        cookies["thamba"] = 2
        cookies["gpp"] = "DBAA"
        cookies["gpp_sid"] = "-1"
        cookie_str = "; ".join([f"{key}={value}" for key, value in cookies.items()])

    headers = {
        "authority": "query1.finance.yahoo.com",
        "method": "GET",
        "path": path,
        "scheme": "https",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Cookie": cookie_str,
        "Dnt": "1",
        "Sec-Ch-Ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    }
    if not cj:
        del headers["Cookie"]

    if run_crumb:
        crumb_url = "https://query2.finance.yahoo.com/v1/test/getcrumb"
        res = requests.get(crumb_url, headers=headers)
        crumb = res.text

        return headers, crumb

    return headers


def download_historical_data_yahoofinance(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
    raw_path: str = None,
    ny_time=False
):
    from_sec = round(from_date.timestamp())
    to_sec = round(to_date.timestamp())
    base_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={from_sec}&period2={to_sec}&interval=1d&events=history&includeAdjustedClose=true"

    _, crumb = get_yahoofinance_download_auth("/v1/test/getcrumb", None, True)
    headers = get_yahoofinance_download_auth(
        f"/v8/finance/chart/{ticker}?period1={from_sec}&amp;period2={to_sec}&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true&crumb={crumb}&formatted=false&region=US&lang=en-US",
    )

    res_url = f"{base_url}&crumb={crumb}&formatted=false&region=US&lang=en-US"
    response = requests.get(res_url, headers=headers, allow_redirects=True)
    if response.ok:
        json_data = response.json()
        df = pd.DataFrame(
            {
                "Date": json_data["chart"]["result"][0]["timestamp"],
                "Open": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                    "open"
                ],
                "High": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                    "high"
                ],
                "Low": json_data["chart"]["result"][0]["indicators"]["quote"][0]["low"],
                "Close": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                    "close"
                ],
                "Adj Close": json_data["chart"]["result"][0]["indicators"]["adjclose"][
                    0
                ]["adjclose"],
                "Volume": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                    "volume"
                ],
            }
        )
        df["Date"] = pd.to_datetime(df["Date"], utc=True, unit="s")
        if ny_time:
            df["Date"] = df["Date"].dt.tz_convert("America/New_York")
        else: 
            df["Date"] = df["Date"].dt.date
        
        if raw_path:
            df.to_excel(raw_path)
        return df

    print(f"yf fetch failed with status: {response.status_code}")
    return pd.DataFrame()


def multi_download_historical_data_yahoofinance(
    tickers: List[str],
    from_date: datetime,
    to_date: datetime,
    data_dump_dir: str = None,
    max_date=False,
    big_wb=False,
) -> Dict[str, pd.DataFrame]:
    from_sec = round(from_date.timestamp())
    to_sec = (
        round(to_date.timestamp())
        if not max_date
        else round(datetime.today().timestamp())
    )

    async def fetch(
        session: aiohttp.ClientSession, url: str, curr_ticker: str, crumb: str
    ) -> pd.DataFrame:
        try:
            headers = get_yahoofinance_download_auth(
                f"/v8/finance/chart/{curr_ticker}?period1={from_sec}&amp;period2={to_sec}&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true&crumb={crumb}&formatted=false&region=US&lang=en-US",
            )
            res_url = f"{url}&crumb={crumb}&formatted=false&region=US&lang=en-US"
            async with session.get(res_url, headers=headers) as response:
                if response.ok:
                    json_data = await response.json()
                    df = pd.DataFrame(
                        {
                            "Date": json_data["chart"]["result"][0]["timestamp"],
                            "Open": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                                "open"
                            ],
                            "High": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                                "high"
                            ],
                            "Low": json_data["chart"]["result"][0]["indicators"]["quote"][0]["low"],
                            "Close": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                                "close"
                            ],
                            "Adj Close": json_data["chart"]["result"][0]["indicators"]["adjclose"][
                                0
                            ]["adjclose"],
                            "Volume": json_data["chart"]["result"][0]["indicators"]["quote"][0][
                                "volume"
                            ],
                        }
                    )
                    df["Date"] = pd.to_datetime(df["Date"], utc=True, unit="s")
                    df["Date"] = df["Date"].dt.tz_convert("America/New_York")
                    df["Date"] = df["Date"].dt.tz_localize(None)
                    df["Date"] = df["Date"].dt.date
                    if data_dump_dir:
                        df.to_excel(fr"{data_dump_dir}\{curr_ticker}.xlsx", index=False)
                    return df 
                else:
                    raise Exception(f"Bad Status: {response.status} - on {curr_ticker}")
        except Exception as e:
            print(e)
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            )

    async def get_promises(session: aiohttp.ClientSession):
        tasks = []
        _, crumb = get_yahoofinance_download_auth("/v1/test/getcrumb", None, True)
        for ticker in tickers:
            if max_date:
                try:
                    headers = get_yahoofinance_download_auth(
                        f"v8/finance/chart/{ticker}?formatted=true&crumb={crumb}&lang=en-US&region=US&includeAdjustedClose=true&corsDomain=finance.yahoo.com",
                    )
                    first_trade_date_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?formatted=true&crumb={crumb}&lang=en-US&region=US&includeAdjustedClose=true&corsDomain=finance.yahoo.com"
                    first_trade_date = requests.get(
                        first_trade_date_url, headers=headers
                    ).json()["chart"]["result"][0]["meta"]["firstTradeDate"]
                    from_sec = round(first_trade_date)
                except Exception as e:
                    # print("First Trade Date Error", e)
                    from_sec = round(from_date.timestamp())
            else:
                from_sec = round(from_date.timestamp())

            curr_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?period1={from_sec}&period2={to_sec}&interval=1d&events=history&includeAdjustedClose=true"
            task = fetch(session, curr_url, ticker, crumb)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def run_fetch_all() -> List[pd.DataFrame]:
        async with aiohttp.ClientSession() as session:
            all_data = await get_promises(session)
            return all_data

    dfs = asyncio.run(run_fetch_all())

    if big_wb:
        tickers_str = str.join("_", [str(x) for x in tickers])
        wb_file_name = f"{data_dump_dir}/{tickers_str}_yahoofin_historical_data.xlsx"
        with pd.ExcelWriter(wb_file_name) as writer:
            for i, df in enumerate(dfs, 0):
                try:
                    df.drop("Unnamed: 0", axis=1, inplace=True)
                except:
                    pass
                df.to_excel(writer, sheet_name=f"{tickers[i]}", index=False)

    return dict(zip(tickers, dfs))


def get_yahoofinance_data_file_path_by_ticker(ticker: str, cj: http.cookiejar = None):
    dir = f"{os.path.abspath('')}/yahoofinance"
    files = sorted(
        os.listdir(dir),
    )
    tickers_with_data = [x for x in files if x.split("_")[0].lower() == ticker.lower()]

    if len(tickers_with_data) == 0:
        from_date = datetime.datetime(2023, 1, 1)
        to_date = datetime.datetime.today()
        download_historical_data_yahoofinance(ticker, from_date, to_date, dir, cj)

    return f"{dir}/{ticker}_yahoofin_historical_data.xlsx"


def get_option_expiration_dates_yahoofinance(
    ticker: str, cj: http.cookiejar = None
) -> List[int]:
    _, crumb = get_yahoofinance_download_auth("/v1/test/getcrumb", cj, True)
    headers = get_yahoofinance_download_auth(
        f"/v7/finance/options/{ticker}?formatted=true&crumb={crumb}&lang=en-US&region=US&date=1&straddle=true&corsDomain=finance.yahoo.com",
        cj,
    )
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}?formatted=true&crumb=wZdpBzDeWLv&lang=en-US&region=US&date=1&straddle=true&corsDomain=finance.yahoo.com"

    try:
        res = requests.get(url, headers=headers)
        json = res.json()
        return json["optionChain"]["result"][0]["expirationDates"]
    except Exception as e:
        print(e)
        return []


def safe_get(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


def empty_option_data_yahoofinance() -> dict[str, None]:
    return {
        "percentChange": None,
        "openInterest": None,
        "strike": None,
        "change": None,
        "inTheMoney": None,
        "impliedVolatility": None,
        "volume": None,
        "ask": None,
        "contractSymbol": None,
        "lastTradeDate": None,
        "expiration": None,
        "currency": None,
        "contractSize": None,
        "bid": None,
        "lastPrice": None,
    }


def get_options_chain_yahoofinance(
    ticker: str,
    raw_path: str,
    cj: http.cookiejar = None,
    big_wb=False,
) -> Dict[date, pd.DataFrame]:
    async def fetch(
        session: aiohttp.ClientSession, url: str, ex_date: int, crumb: str
    ) -> pd.DataFrame:
        try:
            headers = get_yahoofinance_download_auth(
                f"/v7/finance/options/{ticker}?formatted=true&crumb={crumb}&lang=en-US&region=US&date={ex_date}&straddle=true&corsDomain=finance.yahoo.com",
                cj,
            )
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    # straddle schema
                    """
                    type Straddle = {
                        stike: Option
                    }
                    type Option = {
                        call: OptionInfo
                        puts OptionInfo
                    }
                    type OptionInfo = {
                        "openInterest": int
                        "strike": int
                        "change": int
                        "inTheMoney": bool
                        "impliedVolatility": int
                        "volume": int
                        "ask": int
                        "contractSymbol": str
                        "lastTradeDate": int
                        "expiration": int
                        "currency": int
                        "contractSize": int
                        "bid": int
                        "lastPrice": int
                    }
                    """
                    json = await response.json()
                    data = json["optionChain"]["result"][0]["options"][0]["straddles"]

                    straddle = {}
                    for option in data:
                        strike_price = option["strike"]["raw"]

                        call_raw = safe_get(option, "call")
                        call_raw_dict = call_raw.items() if call_raw else None
                        call = (
                            {
                                key: (
                                    value.get("raw", value)
                                    if isinstance(value, dict)
                                    else value
                                )
                                for key, value in call_raw_dict
                                if call_raw_dict
                            }
                            if call_raw_dict
                            else empty_option_data_yahoofinance()
                        )

                        put_raw = safe_get(option, "put")
                        put_raw_dict = put_raw.items() if put_raw else None
                        put = (
                            {
                                key: (
                                    value.get("raw", value)
                                    if isinstance(value, dict)
                                    else value
                                )
                                for key, value in put_raw_dict
                                if put_raw_dict
                            }
                            if put_raw_dict
                            else empty_option_data_yahoofinance()
                        )

                        straddle[strike_price] = {
                            "call": call,
                            "put": put,
                        }

                    return straddle, ex_date

                else:
                    raise Exception(f"Bad Status: {response.status}")

        except Exception as e:
            print(e)
            return {}

    async def get_promises(session: aiohttp.ClientSession):
        # Option Chain Schema
        """
        type OptionChain = {
            date (int epoch): Straddle
        }
        """
        _, crumb = get_yahoofinance_download_auth("/v1/test/getcrumb", cj, True)
        exp_dates = get_option_expiration_dates_yahoofinance(ticker, cj)
        tasks = []
        for exp_date in exp_dates:
            curr_url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}?formatted=true&crumb={crumb}&lang=en-US&region=US&date={exp_date}&straddle=true&corsDomain=finance.yahoo.com"
            task = fetch(session, curr_url, exp_date, crumb)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    """
    return List[Tuple(OptionChain, ExpDate)]
    """

    async def run_fetch_all():
        async with aiohttp.ClientSession() as session:
            all_data = await get_promises(session)
            return all_data

    results = asyncio.run(run_fetch_all())
    dfs = {}
    for_big_wb = []

    try:
        for options_data, exp_date in results:
            strike_prices = list(options_data.keys())
            temp = []
            for strike in strike_prices:
                call_raw_dict = options_data[strike]["call"]
                put_raw_dict = options_data[strike]["put"]

                call_dict = {
                    f"{key}_call": value
                    for key, value in call_raw_dict.items()
                    if key != "strike"
                }
                put_dict = {
                    f"{key}_put": value
                    for key, value in put_raw_dict.items()
                    if key != "strike"
                }
                curr_strike_straddle_dict = {**call_dict, **put_dict}
                curr_strike_straddle_dict["strike"] = (
                    call_raw_dict["strike"]
                    if call_raw_dict["strike"]
                    else put_raw_dict["strike"]
                )

                print(curr_strike_straddle_dict)
                temp.append(curr_strike_straddle_dict)

                if big_wb:
                    copy_curr_strike_straddle_dict = curr_strike_straddle_dict.copy()
                    copy_curr_strike_straddle_dict["expirationDate"] = exp_date
                    for_big_wb.append(copy_curr_strike_straddle_dict)

            curr_exp_straddle_df = pd.DataFrame(temp)
            curr_exp_straddle_df.set_index("strike")
            dfs[exp_date] = curr_exp_straddle_df

        with pd.ExcelWriter(raw_path, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False)
            for exp_date, df in dfs.items():
                cols = list(df.columns)
                last_price_index = cols.index("lastPrice_call")
                cols.remove("strike")
                cols.insert(last_price_index + 1, "strike")
                df = df[cols]
                df.to_excel(
                    writer,
                    sheet_name=f"{datetime.fromtimestamp(exp_date).strftime('%m-%d-%Y')}",
                    index=False,
                )

            if big_wb:
                big_df = pd.DataFrame(for_big_wb)
                big_df.to_excel(writer, sheet_name="all", index=False)

    except Exception as e:
        print(e)

    return dfs


# def get_futures_chain(
#     ticker: str, raw_path: str, cj: http.cookiejar = None, big_wb=False
# ):

#     pass


if __name__ == "__main__":
    t0 = time.time()

    from_date = datetime(2023, 1, 1)
    to_date = datetime.today()

    natgas = ["BOIL", "UNG", "KOLD"]
    rates = ["SVOL", "PFIX", "TUA", "TYA", "MTBA", "^MOVE", "^VIX", "VMBS", "^GSPC"]
    dict = multi_download_historical_data_yahoofinance(
        rates,
        from_date,
        to_date,
        r"C:\Users\chris\rates_trading\data\yahoofinance",
        max_date=True,
    )
    print(dict)

    t1 = time.time()
    print("\033[94m {}\033[00m".format(t1 - t0), " seconds")
