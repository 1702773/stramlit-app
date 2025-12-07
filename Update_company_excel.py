import pandas as pd

TWSE_CSV = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
TPEX_CSV = "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"

twse = pd.read_csv(TWSE_CSV)
tpex = pd.read_csv(TPEX_CSV)
#print(tpex)
#exit()
# 只留三欄，並排序
twse = twse[["公司代號","公司名稱"]].sort_values("公司代號")
tpex = tpex[["公司代號","公司名稱"]].sort_values("公司代號")

twse.to_csv("twse_listed.csv", index=False, encoding="utf-8-sig")
tpex.to_csv("tpex_otc.csv", index=False, encoding="utf-8-sig")

both = pd.concat([twse.assign(市場="上市(TWSE)"),
                  tpex.assign(市場="上櫃(TPEx)")], ignore_index=True)
both.to_csv("tw_all_listed_otc.csv", index=False, encoding="utf-8-sig")

print(f"TWSE 上市家數 = {len(twse)}，TPEx 上櫃家數 = {len(tpex)}")
