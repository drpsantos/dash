import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
st.title('5-Minute Data Comparison')
start_date = st.date_input('Start Date of Data Set',datetime.date(2021,6,26))
end_date = st.date_input('End Date of Data Set',datetime.date(2021,12,15))

@st.cache()
def loadWESMData(url):
    minData = pd.read_excel(url,header=1)
    minData.rename(columns={'Unnamed: 0':'ts'},inplace=True)
    minData['ts_day'] = 0.0
    for i in range(0,len(minData)):
        minData['ts_day'].iloc[i] = (minData['ts'].iloc[i]).date()
    minData['CLUZ'] = minData['CLUZ']/1000.0
    minData.CLUZ.loc[minData.CLUZ > 32.0] = 32.0
    return minData
  
@st.cache()
def loadModelData(url):
    hourlyData = pd.read_csv(url)
    l = (pd.DataFrame(columns=['NULL'],
                    index=pd.date_range('2021-01-01 00:00:00', '2022-01-01 00:00:00',
                                        freq='60T'))
        .index.strftime('%Y-%m-%d %H:%M:%S')
        .to_list()
    )
    data2021 = pd.DataFrame(columns=['ts','WESM Rate','ts_day'],index=range(0,8760))
    data2021['ts'] = pd.date_range('2021-01-01 00:00:00', '2021-12-31 23:00:00',
                                        freq='60T')

    for i in range(0,len(data2021)):
        #data2021['ts'].iloc[i] = l[i].strftime()
        data2021['WESM Rate'].iloc[i] = float(hourlyData['2021'].iloc[i])
        data2021['ts_day'].iloc[i] = (data2021['ts'].iloc[i]).strftime('%Y-%m-%d')
    data2021['WESM Rate'] = pd.to_numeric(data2021['WESM Rate'])

    return data2021

minData = loadWESMData("ref/GWAP 20210626 to 20211215.xlsx")
data2021 = loadModelData('ref/210104 RC WESM Forecast Base Case.csv')

fig = go.Figure()
fig.add_trace(go.Scatter(x=minData.iloc[:,0],y=minData['CLUZ'],name='5 Min Historical Data',line_width=1))
fig.add_trace(go.Bar(x=data2021['ts'],y=data2021['WESM Rate'],name='RC WESM Base Case'))
fig.update_layout(
    legend=dict(orientation='h',x=0.5,xanchor='center',y=1.15),
    xaxis_range = [start_date,end_date],
    title=dict(text='Actual vs. Modeled WESM Data',x=0.5),
    xaxis_title = 'Date',
    yaxis_title = 'PHP/kWh'
)
st.plotly_chart(fig)
col1, col2,col3 = st.columns(3)
col1.subheader('User Input')
battery_kwh = col1.number_input('Battery Capacity (kWh)',8480)
battery_kw = col1.number_input('Battery Hourly Discharge (kW)',2500)
case1_interval = col1.number_input('RC Data Ineterval (mins.)', 60)
case2_interval = col1.number_input('WESM Market Interval (mins.)',5)
acinv_efficiency = col1.number_input('AC Inverter Efficiency',0.93)
dod_allowed = col1.number_input('Maximum Depth of Discharge',0.9)

col2.subheader('Discharging')
dkwh = battery_kwh*dod_allowed*acinv_efficiency
discharge_kwh = col2.number_input('Battery Discharge Capacity (kWh)',min_value=dkwh,max_value=dkwh,value=dkwh)
dtime = dkwh/battery_kw
discharge_time = col2.number_input('Battery Discharge Hours',dtime)
dintv_wesm = col2.number_input('# of Discharging Intervals for WESM',(discharge_time*60)/case2_interval)
dcchargeint_wesm = col2.number_input('Disharge per Interval (kWh)',discharge_kwh/dintv_wesm)

col3.subheader('Charging')
ckwh = dkwh/acinv_efficiency
charge_kwh = col3.number_input('Battery Charge Capacity (kWh)',min_value=ckwh,max_value=ckwh,value=ckwh)
ctime = ckwh/battery_kw
charge_time = col3.number_input('Battery Charge Hours',ctime)
cintv_wesm = col3.number_input('# of Charging Intervals for WESM',(charge_time*60)/case2_interval)
cchargeint_wesm = col3.number_input('Charge per Interval (kWh)',charge_kwh/cintv_wesm)

@st.cache(suppress_st_warning=True)
def getData(discharge_time,dintv_wesm,dcchargeint_wesm,charge_time,cintv_wesm,cchargeint_wesm,start_date,end_date):
    moneyCalc = pd.DataFrame(columns=['ts','min_charge','min_discharge','min_profit','hour_charge','hour_discharge','hour_profit'],index=range(0,int((end_date-start_date).days)+1))
    moneyCalc['ts'] = pd.date_range(start_date, end_date,freq='D')
    moneyCalc['ts'] = pd.to_datetime(moneyCalc['ts']).dt.date
    #st.write(moneyCalc.dtypes['ts'])

    this_date = start_date
    delta = datetime.timedelta(days=1)

    while this_date<=end_date:
        cut1 = data2021.loc[data2021['ts_day']==str(this_date)]
        cut2 = minData.loc[minData['ts_day']==this_date]
        cut2 = cut2[cut2.CLUZ>0]

        cut1top = cut1.nlargest(int(np.ceil(discharge_time)),'WESM Rate',keep='all')
        cut1bot = cut1.nsmallest(int(np.ceil(charge_time)),'WESM Rate',keep='all')
        cut2top = cut2.nlargest(int(np.ceil(dintv_wesm)),'CLUZ',keep='all')
        cut2bot = cut2.nsmallest(int(np.ceil(cintv_wesm)),'CLUZ',keep='all')

        min_charge = (np.floor(cintv_wesm)*sum(cut2bot['CLUZ'].iloc[:-1]))+((cintv_wesm%1)*cut2bot['CLUZ'].iloc[-1])
        min_discharge = (np.floor(dintv_wesm)*sum(cut2top['CLUZ'].iloc[:-1]))+((dintv_wesm%1)*cut2top['CLUZ'].iloc[-1])
        min_profit = min_discharge-min_charge

        hour_charge = (np.floor(charge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+(((charge_time%1)*battery_kw)*cut1bot['WESM Rate'].iloc[-1])
        hour_discharge = (np.floor(discharge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+((cintv_wesm%1)*battery_kw*cut1bot['WESM Rate'].iloc[-1])
        hour_profit = hour_discharge-hour_charge

        moneyCalc.min_charge.loc[moneyCalc.ts == this_date] = min_charge
        moneyCalc.min_discharge.loc[moneyCalc.ts == this_date] = min_discharge
        moneyCalc.min_profit.loc[moneyCalc.ts == this_date] = min_profit
        moneyCalc.hour_charge.loc[moneyCalc.ts == this_date] = hour_charge
        moneyCalc.hour_discharge.loc[moneyCalc.ts == this_date] = hour_discharge
        moneyCalc.hour_profit.loc[moneyCalc.ts == this_date] = hour_profit
        
        this_date += delta

    #st.write(moneyCalc.head(5))
    return moneyCalc

calcData = getData(discharge_time,dintv_wesm,dcchargeint_wesm,charge_time,cintv_wesm,cchargeint_wesm,start_date,end_date)

st.subheader('Timeframe Analysis')

tafig = make_subplots(rows=2,cols=1,subplot_titles=['5 Minute WESM Data','RC WESM 2021'],shared_xaxes=True,vertical_spacing=0.08)
tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.min_charge,name='Charging Cost',legendgroup='group1'),row=1,col=1)
tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.min_discharge,name='Discharging Income',legendgroup='group1'),row=1,col=1)
tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.min_profit,name='Profit',legendgroup='group1'),row=1,col=1)

tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.hour_charge,name='Charging Cost',legendgroup='group2'),row=2,col=1)
tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.hour_discharge,name='Discharging Income',legendgroup='group2'),row=2,col=1)
tafig.add_trace(go.Scatter(x=calcData.ts,y=calcData.hour_profit,name='Profit',legendgroup='group2'),row=2,col=1)

tafig.update_yaxes(title_text='PhP/kWh',row=1,col=1)
tafig.update_yaxes(title_text='PhP/kWh',row=2,col=1)

st.plotly_chart(tafig)

s2col1,s2col2 = st.columns(2)
s2col1.subheader('5-Minute WESM')
s2col1.number_input('Total Charging Cost (PhP)',round(sum(calcData.min_charge),2))
s2col1.number_input('Total Discharging Income (PhP)',round(sum(calcData.min_discharge),2))
s2col1.number_input('Total Profit (PhP)',round(sum(calcData.min_profit),2))

s2col2.subheader('RC WESM 2021')
s2col2.number_input('Total Charging Cost (PhP)',round(sum(calcData.hour_charge),2))
s2col2.number_input('Total Discharging Income (PhP)',round(sum(calcData.hour_discharge),2))
s2col2.number_input('Total Profit (PhP)',round(sum(calcData.hour_profit),2))


st.subheader('One-day Analysis')
this_date = st.date_input('Select Date for Review',datetime.date(2021,6,26))
cut1 = data2021.loc[data2021['ts_day']==str(this_date)]
cut2 = minData.loc[minData['ts_day']==this_date]
cut2 = cut2[cut2.CLUZ > 0.0]
cut1top = cut1.nlargest(int(np.ceil(discharge_time)),'WESM Rate',keep='all')
cut1bot = cut1.nsmallest(int(np.ceil(charge_time)),'WESM Rate',keep='all')
cut2top = cut2.nlargest(int(np.ceil(dintv_wesm)),'CLUZ',keep='all')
cut2bot = cut2.nsmallest(int(np.ceil(cintv_wesm)),'CLUZ',keep='all')

cut3 = minData.loc[minData['ts_day']==this_date]
cut3 = cut3[cut3.CLUZ > 0.0]
cut3top = cut3.nlargest(int(np.ceil(dintv_wesm)),'CLUZ',keep='all')
cut3bot = cut3.nsmallest(int(np.ceil(cintv_wesm)),'CLUZ',keep='all')

min_charge = (np.floor(cintv_wesm)*sum(cut2bot['CLUZ'].iloc[:-1]))+((cintv_wesm%1)*cut2bot['CLUZ'].iloc[-1])
min_discharge = (np.floor(dintv_wesm)*sum(cut2top['CLUZ'].iloc[:-1]))+((dintv_wesm%1)*cut2top['CLUZ'].iloc[-1])
min_profit = min_discharge-min_charge

hour_charge = (np.floor(charge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+(((charge_time%1)*battery_kw)*cut1bot['WESM Rate'].iloc[-1])
hour_discharge = (np.floor(discharge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+((cintv_wesm%1)*battery_kw*cut1bot['WESM Rate'].iloc[-1])
hour_profit = hour_discharge-hour_charge

min_charge2 = (np.floor(cintv_wesm)*sum(cut3bot['CLUZ'].iloc[:-1]))+((cintv_wesm%1)*cut3bot['CLUZ'].iloc[-1])
min_discharge2 = (np.floor(dintv_wesm)*sum(cut3top['CLUZ'].iloc[:-1]))+((dintv_wesm%1)*cut3top['CLUZ'].iloc[-1])
min_profit2 = min_discharge2-min_charge2

# st.write(cut1bot['WESM Rate'])
# st.write(charge_time,np.floor(charge_time),round(charge_time%1,2))
# st.write(battery_kwh)
# st.write(sum(cut1bot['WESM Rate'].iloc[:-1]),cut1bot['WESM Rate'].iloc[-1])
# st.write((np.floor(charge_time)*battery_kw)*sum(cut1bot['WESM Rate'].iloc[:-1]))
# st.write(((charge_time%1)*battery_kw)*cut1bot['WESM Rate'].iloc[-1])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=cut2['ts'],y=cut2['CLUZ'],name='5 Minute Historical Data',legendgroup='group1',line_width=1))
fig2.add_trace(go.Scatter(x=cut1['ts'],y=cut1['WESM Rate'],name='RC WESM 2021',legendgroup='group2',line_width = 1))
fig2.add_trace(go.Scatter(x=cut2top['ts'],y=cut2top['CLUZ'],name='Top '+str(int(np.ceil(dintv_wesm)))+' Intervals',mode='markers',legendgroup='group1'))
fig2.add_trace(go.Scatter(x=cut2bot['ts'],y=cut2bot['CLUZ'],name='Bottom '+str(int(np.ceil(cintv_wesm)))+' Intervals',mode='markers',legendgroup='group1'))
fig2.add_trace(go.Scatter(x=cut1top['ts'],y=cut1top['WESM Rate'],name='Top '+str(int(np.ceil(discharge_time)))+' intervals',legendgroup='group2',mode='markers'))
fig2.add_trace(go.Scatter(x=cut1bot['ts'],y=cut1bot['WESM Rate'],name='Top '+str(int(np.ceil(charge_time)))+' intervals',legendgroup='group2',mode='markers'))
fig2.add_trace(go.Scatter(x=cut3bot['ts'],y=cut3bot['CLUZ'],name='Bottom '+str(int(np.ceil(cintv_wesm)))+' Intervals CASE 2',mode='markers',legendgroup='group1'))
fig2.update_layout(
    title='Data for '+str(this_date),
    yaxis_title = ('PhP/kWh Rate'),
    xaxis_title = ('Time of Day')
)
st.plotly_chart(fig2)


s3col1,s3col3 = st.columns(2)
s3col1.subheader('5-Minute WESM')
s3col1.number_input('Charging Cost (PhP)',round(min_charge,2),round(min_charge,2),round(min_charge,2))
s3col1.number_input('Discharging Income (PhP)',round(min_discharge,2),round(min_discharge,2),round(min_discharge,2))
s3col1.number_input('Profit (PhP)', round(min_profit,2), round(min_profit,2), round(min_profit,2))

# s3col2.subheader('Case 2')
# s3col2.number_input('Charging Cost (PhP)',round(min_charge2,2),round(min_charge2,2),round(min_charge2,2))
# s3col2.number_input('Discharging Income (PhP)',round(min_discharge2,2),round(min_discharge2,2),round(min_discharge2,2),key=123)
# s3col2.number_input('Profit (PhP)', round(min_profit2,2), round(min_profit2,2), round(min_profit2,2))
# s3col2.number_input('Profit Difference (Case 2 - Case 1)',round(min_profit2-min_profit,2))

s3col3.subheader('RC WESM 2021')
s3col3.number_input('Charging Cost (PhP)',round(hour_charge,2),round(hour_charge,2),round(hour_charge,2))
s3col3.number_input('Discharging Income (PhP)',round(hour_discharge,2),round(hour_discharge,2),round(hour_discharge,2))
s3col3.number_input('Profit (PhP)', round(hour_profit,2), round(hour_profit,2), round(hour_profit,2))

st.title('RC WESM Rates Analaysis')
st.text('Analysis of BESS performance given RC WESM 2021 Data')

start_date_RC = datetime.date(2021,1,1)
end_date_RC = datetime.date (2021,12,31)

@st.cache(suppress_st_warning=True)
def getRCData(discharge_time,charge_time,start_date_RC,end_date_RC):
    moneyCalc = pd.DataFrame(columns=['ts','hour_charge','hour_discharge','hour_profit'],index=range(0,int((end_date_RC-start_date_RC).days)+1))
    moneyCalc['ts'] = pd.date_range(start_date_RC, end_date_RC,freq='D')
    moneyCalc['ts'] = pd.to_datetime(moneyCalc['ts']).dt.date
    #st.write(moneyCalc.dtypes['ts'])

    this_date = start_date_RC
    delta = datetime.timedelta(days=1)

    while this_date<=end_date_RC:
        cut1 = data2021.loc[data2021['ts_day']==str(this_date)]

        cut1top = cut1.nlargest(int(np.ceil(discharge_time)),'WESM Rate',keep='all')
        cut1bot = cut1.nsmallest(int(np.ceil(charge_time)),'WESM Rate',keep='all')

        hour_charge = (np.floor(charge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+(((charge_time%1)*battery_kw)*cut1bot['WESM Rate'].iloc[-1])
        hour_discharge = (np.floor(discharge_time)*battery_kw*sum(cut1bot['WESM Rate'].iloc[:-1]))+((cintv_wesm%1)*battery_kw*cut1bot['WESM Rate'].iloc[-1])
        hour_profit = hour_discharge-hour_charge

        moneyCalc.hour_charge.loc[moneyCalc.ts == this_date] = hour_charge
        moneyCalc.hour_discharge.loc[moneyCalc.ts == this_date] = hour_discharge
        moneyCalc.hour_profit.loc[moneyCalc.ts == this_date] = hour_profit
        
        this_date += delta

    #st.write(moneyCalc.head(5))
    moneyCalc['ts_month'] = pd.DatetimeIndex(moneyCalc['ts']).month
    return moneyCalc

# value = 1.0
# st.write(data2021.loc[data2021['WESM Rate']<=value])
# st.write(data2021.loc[data2021['WESM Rate']<=value].shape)
RC_data = getRCData(discharge_time, charge_time, start_date_RC, end_date_RC)
#st.write(RC_data.loc[RC_data.ts_month == 1])
months = 12
RC_monthly = pd.DataFrame(columns=['month','charge','discharge','profit'],index=range(0,months))
for i in range(0,months):
    datetime_object =     datetime.datetime.strptime(str(i+1), "%m")
    RC_monthly['month'].iloc[i] = datetime_object.strftime("%B")
    RC_monthly['charge'].iloc[i] = RC_data['hour_charge'].loc[RC_data.ts_month == (i+1)].sum()
    RC_monthly['discharge'].iloc[i] = RC_data['hour_discharge'].loc[RC_data.ts_month == (i+1)].sum()
    RC_monthly['profit'].iloc[i] = RC_data['hour_profit'].loc[RC_data.ts_month == (i+1)].sum()

st.dataframe(RC_monthly,height=500)

st.title('211208 Code Analysis')
@st.cache
def runOldCode(discharge_time,charge_time):
    moneyCalc = pd.DataFrame(columns=['ts','battery_state','hour_charge','hour_discharge'],index=range(0,int((end_date_RC-start_date_RC).days)+1))
    moneyCalc['ts'] = pd.date_range(start_date_RC, end_date_RC,freq='D')
    moneyCalc['ts'] = pd.to_datetime(moneyCalc['ts']).dt.date

    partial_charge_rate = pcr
    partial_discharge_rate = pdr
    this_year = start_year + year_iter
    charge_time = charge_int
    discharge_time = discharge_int
    charge_hours = int(np.ceil(charge_time+partial_charge_rate))
    discharge_hours = int(np.ceil(discharge_time+partial_discharge_rate))
    bs_charge = 0
    bs_pcharge = 0
    bs_discharge = 0
    bs_pdischarge=0
    charge_cycles = 0
    discharge_cycles = 0
    income = 0
    cost = 0
    print(WESM_this_year)
    battery_state = np.zeros(shape=8760, dtype=int)
    while(i!=len(battery_state)):
        try: 
            charge_window = WESM_this_year[i:i+16]
            print(charge_window)
            cindex = (charge_window).argsort()[:charge_hours]
            print(cindex)
            for j in range(0,len(cindex)):
                this_index = i+cindex[j]
                if j == (len(cindex)-1) and partial_charge_rate>0.0:
                    battery_state[this_index] = 2
                    bs_pcharge = bs_pcharge + 1
                    cost = cost+(WESM_this_year[this_index]*partial_charge_rate*battery_kw)
                else:
                    battery_state[this_index] = 1 # Charging
                    bs_charge = bs_charge+1
                    cost = cost + (WESM_this_year[this_index]*battery_kw)
            charge_cycles = charge_cycles + 1

            print(battery_state[0:24])
            i = i+np.amax(cindex)+1
            print(i)

            discharge_window = WESM_this_year[i:i+activity_interval]
            print(discharge_window)
            dindex = (-discharge_window).argsort()[:discharge_hours]
            print(dindex)
            for j in range(0,len(dindex)):
                this_index = i+dindex[j]
                if j == (len(dindex)-1) and partial_discharge_rate>0.0:
                    battery_state[this_index] = 4
                    bs_pdischarge = bs_pdischarge + 1
                    income = income+(WESM_this_year[this_index]*partial_discharge_rate*battery_kw)
                else:
                    battery_state[this_index] = 3 # Discharging
                    bs_discharge = bs_discharge + 1
                    income = income + (WESM_this_year[this_index]*battery_kw)
            print(battery_state[0:24])
            discharge_cycles = discharge_cycles+1
            i = i+np.amax(dindex)+1
        #print(i)
        except Exception as e:
            #print(i,'//',e)
            break
    count_hours = np.bincount(battery_state)
    hours_charging = bs_charge+ partial_charge_rate*bs_pcharge
    hours_discharging = bs_discharge + partial_discharge_rate*bs_pdischarge
    hours_idle = 8760 - (hours_charging+hours_discharging)
    profit = income - cost
    results = np.array([this_year,(charge_time+partial_charge_rate),(discharge_time+partial_discharge_rate),activity_interval,hours_charging, hours_discharging,hours_idle,charge_cycles,discharge_cycles,round(cost,2),round(income,2),round(profit,2)])
    return results
