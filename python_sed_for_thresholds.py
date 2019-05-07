import pandas as pd
import re
import glob

a = glob.glob(r'output/cost*__q1_0.[0-9]*__q2_0.[0-9]*.tsv')

q1, q2, k0Mean, k0Upper, k0Lower, k1Mean, k1Upper, k1Lower = [], [], [], [], [], [], [], []
for i in a:
    b = re.compile('q1_(0\.[0-9]+)__q2_(0\.[0-9]+)')

    queue1 = b.search(i).group(1)
    queue2 = b.search(i).group(2)

    q1.append(queue1)
    q2.append(queue2)
    k0Mean.append(pd.read_table(i)['mean0'].tolist()[-1])
    k0Upper.append(pd.read_table(i)['upper0'].tolist()[-1])
    k0Lower.append(pd.read_table(i)['lower0'].tolist()[-1])
    k1Mean.append(pd.read_table(i)['mean1'].tolist()[-1])
    k1Upper.append(pd.read_table(i)['upper1'].tolist()[-1])
    k1Lower.append(pd.read_table(i)['lower1'].tolist()[-1])
odf = pd.DataFrame()
odf['q1'] = q1
odf['q2'] = q2
odf['k0Mean'] = k0Mean
odf['k0Upper'] = k0Upper
odf['k0Lower'] = k0Lower
odf['k02Sigma'] = 0.5*(odf['k0Upper'] - odf['k0Lower'])
odf['k1Mean'] = k1Mean
odf['k1Upper'] = k1Upper
odf['k1Lower'] = k1Lower
odf['k12Sigma'] = 0.5*(odf['k1Upper'] - odf['k1Lower'])

odf.sort_values(by='q1', inplace=True)
odf.to_csv('k_summary.csv', index=False) 

odf