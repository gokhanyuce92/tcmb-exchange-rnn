from evds import evdsAPI

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

api_anahtar = read_file('tcmb_evds_api_key.txt').strip()
evds = evdsAPI(api_anahtar)

usd_alis_df = evds.get_data(['TP.DK.USD.A.YTL'], startdate="03-01-2000", enddate="26-05-2025")
usd_alis_df = usd_alis_df.fillna(method='ffill')

usd_alis_df.to_csv('usd_alis_kurlari.csv', index=False)