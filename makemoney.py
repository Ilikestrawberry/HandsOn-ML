import pandas as pd
import requests
from pykrx import stock
import time
from datetime import datetime, timedelta
import re
from tqdm import tqdm
import math
import numpy as np
import json
    
def per(ticker):
    """
    PER 15이하면 10점 만점
    PER 0 이하 0점

    score: 14 - 4PER/15
    """
    per = stock.get_market_fundamental_by_date(datetime.today()-timedelta(days=15), datetime.today(), ticker).iloc[-1]['PBR']
    score = 14 - 4*per/15
    if score >= 10: score = 10
    elif score <= 0: score = 0

    return score


def pbr(ticker):
    """
    PBR 1.5이하면 10점 만점
    PBR 0 이하 0점

    score: 14 - 4PBR/1.5
    """
    pbr = stock.get_market_fundamental_by_date(datetime.today()-timedelta(days=15), datetime.today(), ticker).iloc[-1]['PBR']
    score = 14 - 4*pbr/1.5
    if score >= 10: score = 10
    elif score <= 0: score = 0

    return score


def debt_ratio(debt, equity):
    """
    부채비율이 100%이하면 10점
    175%~100% 7점
    250%~175% 4점
    """
    debt_ratio = debt/equity

    score = 0
    if debt_ratio <= 1:
        score = 10
    elif 1 < debt_ratio <= 1.75:
        score = 7
    elif 1.75 < debt_ratio <= 2.5:
        score = 4

    return score


def roa_over_20(net_profit, asset):
    """
    직전분기 ROA 20% 이상 10점 만점

    score: 10 - 50*(0.2-ROA)
    """
    roa = net_profit/asset
    score = 10 - 50*(0.2-roa)
    if score >= 10:
        score = 10
    elif score < 0:
        score = 0

    return score


def prr_under15(mkcap, cash_flow):
    """
    PRR 계산
    """
    score = 0
    if cash_flow != 0:
        prr = -mkcap/cash_flow
        if 0 < prr <= 15:
            score = 10
    else:
        pass

    return score


def increse_profit(profit):
    """
    지난 3년간 영업이익 지속적으로 증가
    """
    score = 0
    profit_3y = profit[:3]
    if profit_3y[0] < profit_3y[1] < profit_3y[2]:
        score = 10

    return score


def debt_decrese(debt, equity):
    """
    지난 3년간 부채비율 지속적으로 감소
    """
    score = 0
    debt_3y = (debt/equity)[:3]
    if debt_3y[0] > debt_3y[1] > debt_3y[2]:
        score = 10

    return score


def net_profit(net_profit):
    """
    지난 3년간 당기순이익 0 이상
    """
    net_profit = net_profit[:3]
    check = list(net_profit > 0)
    score = 0
    if check == ([True]*3):
        score = 10

    return score


def cash_flow(cash, debt):
    """
    최근 분기 유동비율 100% 이상
    """
    score = 0
    cf = cash/debt
    if cf >= 1:
        score = 10

    return score


def psr_1(profit, ticker):
    """
    지난 4분기 매출액 기준
    PSR 1 이하 10점
    """
    price = stock.get_market_ohlcv_by_date(datetime.today()-timedelta(days=15), datetime.today(), ticker).iloc[-1]['종가']
    stocks = stock.get_market_cap_by_date(datetime.today()-timedelta(days=15), datetime.today(), ticker).iloc[-1]['상장주식수']
    sale = (profit*100000000)/stocks
    psr = price/sale
    score = 0
    if psr <= 1:
        score = 10

    return score


def convert_ticker(tickers):
    """
    티커 => 회사명
    """
    titles = []
    for ticker in tickers:
        title = stock.get_market_ticker_name(ticker)
        titles.append(title)

    return titles


def make_portfolio(equity, tickers):
    """
    금액과 티커를 입력하면 샤프지수에 비례해서 각 종목당 매수할 개수를 반환
    그리고 그 개수를 기반으로
    최근 3년간 변동성과 수익률을 90일 기준으로 환산
    """
    df = pd.DataFrame()
    kospi = stock.get_index_ohlcv_by_date(datetime.today() - timedelta(days=365*3), datetime.today(), "1001")['종가']

    for ticker in tickers:
        stoc = stock.get_market_ohlcv_by_date(datetime.today() - timedelta(days=365*3), datetime.today(), ticker)['종가']
        title = stock.get_market_ticker_name(ticker)
        df2 = pd.DataFrame({title: stoc})
        df = pd.concat([df, df2], axis=1)
    
    ddf = df.pct_change()

    std = ddf.std()*np.sqrt(90)
    returnn = ddf.mean()*90
    sharpe = returnn/std
    weight = sharpe/sum(sharpe)
    price = df.iloc[-1]
    n_stock = round(weight*equity/price)
    pocket = n_stock*price

    cov = ddf.cov()

    p_return = sum(returnn*n_stock)/sum(n_stock)
    p_vol = np.sqrt(weight.dot(cov).dot(weight)*90)
    p_sharpe = p_return/p_vol

    kospi_return = kospi.pct_change().mean()*90
    kospi_vol = kospi.pct_change().std()*np.sqrt(90)
    kospi_sharpe = kospi_return / kospi_vol

    results = pd.DataFrame({'변동성': std, '수익률': returnn, '샤프지수': sharpe, 'Weight': weight,
                           'Price': price, '# stock': n_stock, 'Pocket': pocket}, index=df.columns)

    print('KOSPI(90일 환산) 수익률: {:.2f}% / 변동성: {:.3f} / 샤프지수: {:.3f}'.format(kospi_return*100, kospi_vol, kospi_sharpe))
    print('포트폴리오(90일 환산) 수익률: {:.2f}% / 변동성: {:.3f} / 샤프지수: {:.3f}'.format(p_return*100, p_vol, p_sharpe))
    return results


def save_score(results):
    results.to_csv('results_{}.csv'.format(datetime.now().strftime("%Y %m %d")))
    print('{}자 저장완료'.format(datetime.now().strftime("%Y %m %d")))



class STOCK_SCORE():
    def __init__(self):
        self.mkcap = pd.concat([stock.get_market_cap_by_ticker(datetime.today(), market='KOSDAQ')['시가총액']/100000000, 
                                stock.get_market_cap_by_ticker(datetime.today(), market='KOSPI')['시가총액']/100000000])
        self.tickers = self.mkcap[self.mkcap >= 1000].index  ## 시총 1000억 이상 기업만
    
    
    def save_df(self):
        annual_df = {}
        quarter_df = {}
        print('재무재표 업데이트 중...')
        for ticker in tqdm(self.tickers):
            try:
                fs_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gicode=A{}&cID=&MenuYn=Y&ReportGB=&NewMenuID=103&stkGb=701'.format(ticker)
                fs_page = requests.get(fs_url)
                fs_tables = pd.read_html(fs_page.text)

                s = list(fs_tables[1].columns)
                m = re.findall('[0-9]{4}/[0-9]{2}',str(s))  ## 분기별

                temp_df = fs_tables[1]
                try:
                    temp_df = temp_df.set_index('IFRS(개별)')
                except KeyError:
                    temp_df = temp_df.set_index('IFRS(연결)')
                temp_df = temp_df[m]
                temp_df = temp_df.loc[['매출액','영업이익','당기순이익']]

                temp_df3 = fs_tables[3]
                try:
                    temp_df3 = temp_df3.set_index('IFRS(개별)')
                except KeyError:
                    temp_df3 = temp_df3.set_index('IFRS(연결)')
                temp_df3 = temp_df3.loc[['자산','부채','자본', '유동자산계산에 참여한 계정 펼치기', '유동부채계산에 참여한 계정 펼치기']]

                temp_df5 = fs_tables[5]
                try:
                    temp_df5 = temp_df5.set_index('IFRS(개별)')
                except KeyError:
                    temp_df5 = temp_df5.set_index('IFRS(연결)')
                temp_df5 = temp_df5.loc[['투자활동으로인한현금흐름']]

                fs_df = pd.concat([temp_df, temp_df3, temp_df5])


                s2 = list(fs_tables[0].columns)
                m2 = re.findall('[0-9]{4}/[0-9]{2}',str(s))  ## 연도별

                temp_df0 = fs_tables[0]
                try:
                    temp_df0 = temp_df0.set_index('IFRS(개별)')
                except KeyError:
                    temp_df0 = temp_df0.set_index('IFRS(연결)')
                temp_df0 = temp_df0.loc[['영업이익', '당기순이익']]

                temp_df2 = fs_tables[2]
                try:
                    temp_df2 = temp_df2.set_index('IFRS(개별)')
                except KeyError:
                    temp_df2 = temp_df2.set_index('IFRS(연결)')
                temp_df2 = temp_df2.loc[['부채','자본']]

                fs_df2 = pd.concat([temp_df0, temp_df2])

                quarter_df[ticker] = fs_df.to_dict()
                annual_df[ticker] = fs_df2.to_dict()
                
            except:
                pass

        with open('quarter_df.json', 'w') as outfile:
            json.dump(quarter_df, outfile, ensure_ascii=False)

        with open('annual_df.json', 'w') as outfile:
            json.dump(annual_df, outfile, ensure_ascii=False)
        
        
    def make_df(self):
        
        results = pd.DataFrame()
        
        with open('annual_df.json', "r") as json_file:
            annual_data = json.loads(json_file.read())
        with open('quarter_df.json', "r") as json_file:
            quarter_data = json.loads(json_file.read())
            
        for ticker in tqdm(self.tickers):
            try:
                
                annual_da = pd.DataFrame(annual_data[ticker])
                quarter_da = pd.DataFrame(quarter_data[ticker])
                
                s = list(quarter_da.columns)
                m = re.findall('[0-9]{4}/[0-9]{2}',str(s))  ## 분기별
                s2 = list(annual_da.columns)
                m2 = re.findall('[0-9]{4}/[0-9]{2}',str(s))  ## 연도별

                results.loc[ticker, 'PBR'] = pbr(ticker)
                time.sleep(0.2)
                results.loc[ticker, 'PER'] = per(ticker)
                results.loc[ticker, 'Debt ratio'] = debt_ratio(quarter_da[m[-1]]['부채'], quarter_da[m[-1]]['자본'])
                results.loc[ticker, 'ROA over20'] = roa_over_20(sum(quarter_da.loc['당기순이익']), quarter_da[m[-1]]['자산'])
                results.loc[ticker, 'PRR under15'] = prr_under15(self.mkcap[ticker], sum(quarter_da.loc['투자활동으로인한현금흐름']))
                results.loc[ticker, 'Cash flow'] = cash_flow(quarter_da[m[-1]]['유동자산계산에 참여한 계정 펼치기'], quarter_da[m[-1]]['유동부채계산에 참여한 계정 펼치기'])
                results.loc[ticker, 'PSR under1'] = psr_1(sum(quarter_da.loc['매출액']), ticker)
                results.loc[ticker, '3y Increse Profit'] = increse_profit(annual_da.loc['영업이익'])
                results.loc[ticker, '3y Decrese Debt'] = debt_decrese(annual_da.loc['부채'], annual_da.loc['자본'])
                results.loc[ticker, '3y Positive Profit'] = net_profit(annual_da.loc['당기순이익'])

            except:
                pass
            
            
        results = results.dropna()
        tickers = results.index
        results.index = convert_ticker(results.index)

        return results, tickers
    
    
    def scoring(self):
        results, tickers = self.make_df()
        results['Score'] = results.sum(axis=1)
        results = results.sort_values(by=['Score'], ascending=False)
        results = results.round(decimals=3)
        print(results['Score'].describe())
        
        return results, tickers