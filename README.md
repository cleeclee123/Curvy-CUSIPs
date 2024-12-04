# Curvy-CUSIPs

Using Open Sourced Data for Rates Trading examples in Python:

- Cash Treasuries: [FedInvest EOD Markings](https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate)
- Swaps/Swaptions: [DTCC SDR](https://pddata.dtcc.com/ppd/cftcdashboard), [BBG SEF](https://data.bloombergsef.com/)
- SOFR OIS Curve: [Eris SOFR Swap Futures FTP](https://files.erisfutures.com/ftp/)
- Economics/Misc: [FRED](https://fred.stlouisfed.org/), [NYFRB](https://markets.newyorkfed.org/static/docs/markets-api.html)

## To get started:

Clone repo:
```
git clone https://github.com/yieldcurvemonkey/Curvy-CUSIPs.git
```

pip install dependencies: 
```
pip install -r requirements.txt
```

cd into scripts dir:
```
cd .\CurvyCUSIPs\scripts
```

Update SOFR OIS Curve DB:
```
python update_sofr_ois._db.py
```

## Checkout examples in `\notebooks`:

### sabr_smile.ipynb

![til](./dump/sabrsmileexample.gif)

### More examples/notebooks coming soon