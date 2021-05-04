# Flask : library utama untuk membuat API
# render_template : agar dapat memberikan respon file html
# request : untuk membaca data yang diterima saat request datang
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
# plotly dan plotly.graph_objs : membuat plot
import plotly
import plotly.graph_objs as go
# pandas : untuk membaca csv dan men-generate dataframe
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

## Joblib untuk Load Model
import joblib

# untuk membuat route
app = Flask(__name__)

###################
## CATEGORY PLOT ##
###################

## IMPORT DATA USING pd.read_csv
# tips = pd.read_csv('./static/tips.csv')

# IMPORT DATA USING pd.read_sql
sqlengine = create_engine('mysql+pymysql://root:RussianRoulette5%@127.0.0.1/flaskapp', pool_recycle=3605)
dbConnection = sqlengine.connect()
engine = sqlengine.raw_connection()
cursor = engine.cursor()
df = pd.read_sql("select * from Melbourne", dbConnection)

# category plot function
def category_plot(
    cat_plot = 'histplot',
    cat_x = 'Price', cat_y = 'Price',
    estimator = 'count', hue = 'Type'):

    # generate dataframe tips.csv
    # tips = pd.read_csv('./static/tips.csv')



    # jika menu yang dipilih adalah histogram
    if cat_plot == 'histplot':
        # siapkan list kosong untuk menampung konfigurasi hist
        data = []
        # generate config histogram dengan mengatur sumbu x dan sumbu y
        for val in df[hue].unique():          # Histogram di plotly itu countplot
            hist = go.Histogram(
                x=df[df[hue]==val][cat_x],
                y=df[df[hue]==val][cat_y],
                histfunc=estimator,
                name=str(val)
            )
            #masukkan ke dalam array
            data.append(hist)
        #tentukan title dari plot yang akan ditampilkan
        title='Histogram'
    elif cat_plot == 'boxplot':
        data = []

        for val in df[hue].unique():
            box = go.Box(
                x=df[df[hue] == val][cat_x], #series
                y=df[df[hue] == val][cat_y],
                name = str(val)
            )
            data.append(box)
        title='Box'
    # menyiapkan config layout tempat plot akan ditampilkan
    # menentukan nama sumbu x dan sumbu y
    if cat_plot == 'histplot':
        layout = go.Layout(
            title=title,
            xaxis=dict(title=cat_x),
            yaxis=dict(title='Count'),
            # boxmode group digunakan berfungsi untuk mengelompokkan box berdasarkan hue
            boxmode = 'group'
        )
    else:
        layout = go.Layout(
            title=title,
            xaxis=dict(title=cat_x),
            yaxis=dict(title=cat_y),
            # boxmode group digunakan berfungsi untuk mengelompokkan box berdasarkan hue
            boxmode = 'group'
        )
    #simpan config plot dan layout pada dictionary
    result = {'data': data, 'layout': layout}

    #json.dumps akan mengenerate plot dan menyimpan hasilnya pada graphjson
    graphJSON = json.dumps(result, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

# akses halaman menuju route '/' untuk men-test
# apakah API sudah running atau belum
@app.route('/')
def index():

    plot = category_plot()
    # dropdown menu
    # kita lihat pada halaman dashboard terdapat menu dropdown
    # terdapat lima menu dropdown, sehingga kita mengirimkan kelima variable di bawah ini
    # kita mengirimnya dalam bentuk list agar mudah mengolahnya di halaman html menggunakan looping
    list_plot = [('histplot', 'Histogram'), ('boxplot', 'Box')]
    list_x = [('Price', 'Price'),('Rooms', 'Bedroom'), ('Bathroom', 'Bathroom'), ('Type', 'Type'), ('Car', 'Car'), ('Regionname', 'Region name'), 
    ('Suburb', 'Suburb'), ('SellerG', 'SellerG'), ('CouncilArea', 'CouncilArea')]
    list_y = [('Price', 'Price')]
    list_est = [('count', 'Count'), ('avg', 'Average'), ('max', 'Max'), ('min', 'Min')]
    list_hue = [('Type', 'Type'), ('Car', 'Car')]

    return render_template(
        # file yang akan menjadi response dari API
        'category.html',
        # plot yang akan ditampilkan
        plot=plot,
        # menu yang akan tampil di dropdown 'Jenis Plot'
        focus_plot='histplot',
        # menu yang akan muncul di dropdown 'sumbu X'
        focus_x='Price',

        # untuk sumbu Y tidak ada, nantinya menu dropdown Y akan di disable
        # karena pada histogram, sumbu Y akan menunjukkan kuantitas data

        # menu yang akan muncul di dropdown 'Estimator'
        focus_estimator='count',
        # menu yang akan tampil di dropdown 'Hue'
        focus_hue='Type',
        # list yang akan digunakan looping untuk membuat dropdown 'Jenis Plot'
        drop_plot= list_plot,
        # list yang akan digunakan looping untuk membuat dropdown 'Sumbu X'
        drop_x= list_x,
        # list yang akan digunakan looping untuk membuat dropdown 'Sumbu Y'
        drop_y= list_y,
        # list yang akan digunakan looping untuk membuat dropdown 'Estimator'
        drop_estimator= list_est,
        # list yang akan digunakan looping untuk membuat dropdown 'Hue'
        drop_hue= list_hue)

# ada dua kondisi di mana kita akan melakukan request terhadap route ini
# pertama saat klik menu tab (Histogram & Box)
# kedua saat mengirim form (saat merubah salah satu dropdown) 
@app.route('/cat_fn/<nav>')
def cat_fn(nav):

    # saat klik menu navigasi
    if nav == 'True':
        cat_plot = 'histplot'
        cat_x = 'Rooms'
        cat_y = 'Price'
        estimator = 'count'
        hue = 'Type'
    
    # saat memilih value dari form
    else:
        cat_plot = request.args.get('cat_plot')
        cat_x = request.args.get('cat_x')
        cat_y = request.args.get('cat_y')
        estimator = request.args.get('estimator')
        hue = request.args.get('hue')

    # Dari boxplot ke histogram akan None
    if estimator == None:
        estimator = 'count'
    
    # Saat estimator == 'count', dropdown menu sumbu Y menjadi disabled dan memberikan nilai None
    if cat_y == None:
        cat_y = 'Price'

    # Dropdown menu
    list_plot = [('histplot', 'Histogram'), ('boxplot', 'Box')]
    list_x = [('Price', 'Price'),('Rooms', 'Bedroom'), ('Bathroom', 'Bathroom'), ('Type', 'Type'), ('Car', 'Car'),
    ('Regionname', 'Region name'), ('Suburb', 'Suburb'), ('SellerG', 'SellerG'), ('CouncilArea', 'CouncilArea')]
    list_y = [('Price', 'Price')]
    list_est = [('count', 'Count'), ('avg', 'Average'), ('max', 'Max'), ('min', 'Min')]
    list_hue = [('Type', 'Type'), ('Car', 'Car')]


    plot = category_plot(cat_plot, cat_x, cat_y, estimator, hue)
    return render_template(
        # file yang akan menjadi response dari API
        'category.html',
        # plot yang akan ditampilkan
        plot=plot,
        # menu yang akan tampil di dropdown 'Jenis Plot'
        focus_plot=cat_plot,
        # menu yang akan muncul di dropdown 'sumbu X'
        focus_x=cat_x,
        focus_y=cat_y,

        # menu yang akan muncul di dropdown 'Estimator'
        focus_estimator=estimator,
        # menu yang akan tampil di dropdown 'Hue'
        focus_hue=hue,
        # list yang akan digunakan looping untuk membuat dropdown 'Jenis Plot'
        drop_plot= list_plot,
        # list yang akan digunakan looping untuk membuat dropdown 'Sumbu X'
        drop_x= list_x,
        # list yang akan digunakan looping untuk membuat dropdown 'Sumbu Y'
        drop_y= list_y,
        # list yang akan digunakan looping untuk membuat dropdown 'Estimator'
        drop_estimator= list_est,
        # list yang akan digunakan looping untuk membuat dropdown 'Hue'
        drop_hue= list_hue
    )

##################
## SCATTER PLOT ##
##################

# scatter plot function
def scatter_plot(cat_x, cat_y, hue):


    data = []

    for val in df[hue].unique():
        scatt = go.Scatter(
            x = df[df[hue] == val][cat_x],
            y = df[df[hue] == val][cat_y],
            mode = 'markers',
            name = str(val)
        )
        data.append(scatt)

    layout = go.Layout(
        title= 'Scatter',
        title_x= 0.5,
        xaxis=dict(title=cat_x),
        yaxis=dict(title=cat_y)
    )

    result = {"data": data, "layout": layout}

    graphJSON = json.dumps(result,cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/scatt_fn')
def scatt_fn():
    cat_x = request.args.get('cat_x')
    cat_y = request.args.get('cat_y')
    hue = request.args.get('hue')

    # WAJIB! default value ketika scatter pertama kali dipanggil
    if cat_x == None and cat_y == None and hue == None:
        cat_x = 'Price'
        cat_y = 'Rooms'
        hue = 'Type'

    # Dropdown menu
    list_x = [('Rooms', 'Bedrooms'), ('Bathroom', 'Bathroom'), ('Regionname', 'Region Name'),('CouncilArea', 'Council Area'),('Price','Price'),
    ('Landsize','Landsize'),('BuildingArea','BuildingArea')]
    list_y = [('Price','Price')]
    list_hue = [('Type', 'Type')]

    plot = scatter_plot(cat_x, cat_y, hue)

    return render_template(
        'scatter.html',
        plot=plot,
        focus_x=cat_x,
        focus_y=cat_y,
        focus_hue=hue,
        drop_x= list_x,
        drop_y= list_y,
        drop_hue= list_hue
    )

##############
## PIE PLOT ##
##############

def pie_plot(hue = 'Type'):
    


    vcounts = df[hue].value_counts()

    labels = []
    values = []

    for item in vcounts.iteritems():
        labels.append(item[0])
        values.append(item[1])
    
    data = [
        go.Pie(
            labels=labels,
            values=values
        )
    ]

    layout = go.Layout(title='Pie', title_x= 0.48)

    result = {'data': data, 'layout': layout}

    graphJSON = json.dumps(result,cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/pie_fn')
def pie_fn():
    hue = request.args.get('hue')

    if hue == None:
        hue = 'Type'

    list_hue = [('Type', 'Type'), ('Car', 'Car'),('Regionname', 'Region Name'),
    ('Method', 'Method')]

    plot = pie_plot(hue)
    return render_template('pie.html', plot=plot, focus_hue=hue, drop_hue= list_hue)


@app.route('/pred_lr')
## Menampilkan Dataset
def pred_lr():
    sqlengine = create_engine('mysql+pymysql://root:RussianRoulette5%@127.0.0.1/flaskapp', pool_recycle=3605)
    engine = sqlengine.raw_connection()
    cursor = engine.cursor()
    cursor.execute("SELECT * FROM Melbourne LIMIT 1000")
    data = cursor.fetchall()
    return render_template('predict.html', data=data)

@app.route('/pred_result', methods=['POST', 'GET'])

def pred_result():
    df1 = pd.read_csv('MelbourneCleanInterpolate.csv')
    if request.method == 'POST':
    ## Untuk Predict
        input = request.form
        
        Suburb = input['Suburb'] 

        Bedroom = int(input['Rooms'])

        Type = str(input['Type'])

        Distance = 0
        if Suburb in list(df['Suburb'].unique()):
            Distance = df.loc[df['Suburb']==Suburb,'Distance'].median()

        Bathroom = int(input['Bathroom'])

        Landsize = int(input['Landsize'])
        
        BuildingArea= int(input['BuildingArea'])
        
        data_baru = [{
        'Rooms': Bedroom,
        'Distance': Distance,
        'Bathroom': Bathroom,
        'Landsize': Landsize,
        'BuildingArea': BuildingArea,
        'Suburb': Suburb,
        'Type' : Type,}]

        percobaan_1 = pd.DataFrame(data_baru, index=[1])

        pred = model.predict(percobaan_1)[0]
        pred = np.exp(pred)
        predX = f"AUD {pred.round():,}"
        # print(f"AUD {pred:,}")

        ## Untuk Isi Data

        Suburb_dt = Suburb

        Bedrooms_dt = Bedroom

        Type_dt = ''
        if Type == 'h':
            Type_dt = 'House'
        elif Type == 't':
            Type_dt = 'Townhouse'
        else:
            Type_dt = 'Unit'

        Distance_dt = Distance

        Bathroom_dt = Bathroom
        
        Landsize_dt = Landsize

        BuildingArea_dt = BuildingArea

        # def combo(x):
        #     return str(x['Rooms']) + " " + str(x['Type']) + " " + str(x['Bathroom']) + " " + str(x['Landsize']) + " " + str(x['BuildingArea'])

        # df['combo_features'] = df.apply(combo, axis=1)
        # cv = CountVectorizer()
        # house_matrix = cv.fit_transform(df['combo_features'])
        # # cos_score = cosine_similarity(house_matrix)

        def index_by_feature(room,bath,types,prices):
            return list(df1.loc[(df1['Rooms']==room) & (df1['Type']==types) & (df1['Bathroom']==bath)& (df1['Price']<=prices),['Price']].sort_values('Price',ascending=False).index)
        
        def recommendation(ind):
            return list(df1.loc[[ind],'Suburb'])[0], list(df1.loc[[ind],'Address'])[0], list(df1.loc[[ind],'Price'])[0], list(df1.loc[[ind],'Distance'])[0],list(df1.loc[[ind],'Landsize'])[0],list(df1.loc[[ind],'BuildingArea'])[0]

        index_feat = index_by_feature(Bedrooms_dt,Bathroom_dt,Type,pred)
        
        def rec(x):
            for i in range(5):
                rec3 = list(x)
                if  type(rec3[i]) == float:
                    rec3[i] = f"{rec3[i]:,}"
                    return rec3
       
        # tester = 'Coba'
        return render_template('result.html',
            Suburbs=Suburb_dt,
            Bedrooms=Bedrooms_dt,
            Type=Type_dt,
            Distance=Distance_dt,
            Bathroom =Bathroom_dt,
            Landsize=Landsize_dt,
            BuildingArea=BuildingArea_dt,
            House_pred = predX,
            recommend1= rec(recommendation(index_feat[0])),
            recommend2= rec(recommendation(index_feat[1])),
            recommend3= rec(recommendation(index_feat[2])),
            recommend4= rec(recommendation(index_feat[3])),
            recommend5= rec(recommendation(index_feat[4])),
            # tester=tester
            )

if __name__ == '__main__':
    ## Me-Load data dari Database
    sqlengine = create_engine('mysql+pymysql://root:RussianRoulette5%@127.0.0.1/flaskapp', pool_recycle=3605)
    dbConnection = sqlengine.connect()
    engine = sqlengine.raw_connection() 
    cursor = engine.cursor()
    df = pd.read_sql("select * from Melbourne", dbConnection) # ganti tips dengan nama table sendiri
    # df = pd.read_csv('tips')
    ## Load Model
    model = joblib.load('MelbourneModel')
    app.run(debug=True)