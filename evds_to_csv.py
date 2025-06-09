from evds import evdsAPI

def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

api_anahtar = read_file('tcmb_evds_api_key.txt').strip()
evds = evdsAPI(api_anahtar)

usd_alis_df = evds.get_data(['TP.DK.USD.A.YTL'], 
                            startdate="31-12-1999", 
                            enddate="05-06-2025")
usd_alis_df = usd_alis_df.fillna(method='ffill')

usd_alis_df.to_csv('usd_alis_kurlari.csv', index=False)