from googletrans import Translator # menterjemahkan data
from textblob import TextBlob # Memberi label 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # melakukan stemming
from flask import Flask, redirect, render_template, request, flash, url_for # fungsi bawaan dari flas
import tweepy # crawling data twitter
import csv # menulis dan membaca file csv
import re # mengecek karakter tertentu. re.sub("angka", "")
from PIL import Image # untuk memanipulasi image. entah buka gambar dll
import urllib.request # menangani hal yang berhubungan dengan url. contoh : mengambil gambar gojek dari server untuk wordcloud
from sklearn.metrics import classification_report # menghasilkan laporan klasifikasi berdasarkan data y testing dan y prediksi
import nltk
from nltk.corpus import stopwords # stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split # membagi data training dan testing
from sklearn.feature_extraction.text import TfidfVectorizer # menghitung nilai tfidf
from sklearn.neighbors import KNeighborsClassifier # menghitung KNN
from scipy.sparse import csr_matrix 
import pandas as pd # mengelola data terstruktur, seperti membaca csv
import numpy as np # untuk mengetahui nilai unik array dan mengembalikan nilai unik yang diurutkan
from sklearn.metrics import accuracy_score # mendapatkan akurasi 
from sklearn.metrics import confusion_matrix # menghitung confusion matrix
from wordcloud import WordCloud # men genenerate wordcloud
import matplotlib.pyplot as plt # membuat visualisasi



app = Flask(__name__)

app.config['SECRET_KEY'] = 'faiza'


analysis_result = []
hasil_pembobotan = []

# membuka library
# tekan ctrl + sorot pada library yang ingin dilihat


def crawling_data(query,  count):
    api_key = "5t4ZehrDahkfTqhQsM9dplmih"
    api_secret_key = "jAIPsxDYCtLg9AA9WYq6E5N1vBxCupo2EqS9xJKPr1MMxV9WCV"
    access_token = "726397221550194688-7omVPL8iy7sbexGjMnGBbPQLvLR8jzX"
    access_token_secret = "lCQnI0Fqs7PricXL0nsXjq2aoKuwFRuWCErPZve8XjIOM"
    
    # auth
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # CSV
    file = open('static/files/Data Mentah Twitter.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    analysis_result.clear()



    # memfilter retweets
    filter = " -filter:retweets"


    # get data
    for tweet in tweepy.Cursor(api.search, q=query + filter, lang="id", tweet_mode="extended").items(int(count)):
        flash('Crawling Berhasil', 'success')
        tweets = [tweet.created_at, tweet.user.screen_name,
                  tweet.full_text.replace('\n', '')]
        writer.writerow(tweets)
        analysis_result.append(tweets)



preprocessing_result = []
normalizad_word_dict = {}
def preprocessing_data():
    global normalizad_word_dict
    file = open('static/files/Data Hasil Preprocessing.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    # buka data normalisasi
    normalizad_word = pd.read_excel("static/files/normalisasi.xlsx")
    
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]
            

   
    
    # data stopword
    listStopword =  set(stopwords.words('indonesian'))
    
    stop_word_pribadi = []
    

    with open('static/files/stopwordspribadi.txt', 'r') as f:
        for line in f:
            stop_word_pribadi.append(line.rstrip())


    with open("static/files/Data Mentah Twitter.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            flash('Berhasil preprocessing data ', 'preprocessing_category')
            # menghapus karakter tidak penting
            clean = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", row[2]).split())
            clean = re.sub("\d+", "", clean)
            clean = re.sub(r"\b[a-zA-Z]\b","", clean)

            # casefold
            casefold = clean.casefold()
            # normaliassi
            normalisasi = proses_normalisasi(casefold)
            # tokenizing
            tokenize = nltk.tokenize.word_tokenize(normalisasi)
            # stopword menggunakan library nltk
            listStopword = stopwords.words('indonesian') + stop_word_pribadi
            stopwordsiappakai = set(listStopword)
            hasil_stopwords = [t for t in tokenize if t not in stopwordsiappakai]
           
                    
            sentence = ' '.join(hasil_stopwords)

            # stemming 
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemming = stemmer.stem(sentence)
        

            

            tweet_asli = row[2].replace(',', '')
            # write csv
            tweets = [tweet_asli, clean, casefold, tokenize, hasil_stopwords, stemming]
            writer.writerow(tweets)
            preprocessing_result.append(tweets)
            flash('Berhasil preprocessing data ', 'success')


labelling_result = []
def labelling_data():
    file = open('static/files/Data Hasil Labelling.csv',
                'w', newline='', encoding='utf-8')
    writer = csv.writer(file)

    translator = Translator()

    with open("static/files/Data Hasil Preprocessing.csv", encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            
            tweet = {}
            try:
                value = translator.translate(row[5], dest='en')
            except:
                print("Terjadi error", flush=True)
            terjemahan = value.text
            data_label = TextBlob(terjemahan)
            if data_label.sentiment.polarity > 0.0:
                tweet['sentiment'] = "Positif"
            elif data_label.sentiment.polarity == 0.0:
                tweet['sentiment'] = "Netral"
            else:
                tweet['sentiment'] = "Negatif"

            label = tweet['sentiment']
        
            # write csv
            tweets = [row[5], label]
            writer.writerow(tweets)
            labelling_result.append(tweets)
            flash('Berhasil labelling data ', 'labelling_category')

def normalized_term(data):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in data]      
def proses_normalisasi(data):
    tokens = nltk.tokenize.word_tokenize(data)
    hasil = normalized_term(tokens)
    kalimat = ' '.join(hasil)
    return kalimat


df= None
df2 = None
akurasi = 0
def klasifikasi_data(nilai_k):
    global df
    global df2
    global akurasi
    data = pd.read_csv('static/files/Data Hasil Labelling.csv')
    tweet_data = data.iloc[:, 0].fillna(' ')
    label_data =  data.iloc[:, 1].fillna(' ')

    
    X_train, X_test, y_train, y_test = train_test_split(tweet_data, label_data, test_size=0.2, train_size=0.8, random_state=42)

    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(tweet_data)
    names = Tfidf_vect.get_feature_names()


    # #  mengubah dokumen ke bentuk matrix menggunakan vocabulary dan dokumen yang sudah difit  
    Train_X_Tfidf= Tfidf_vect.transform(X_train)
    Test_X_Tfidf= Tfidf_vect.transform(X_test)

    df_tfidf = pd.DataFrame(data=csr_matrix.todense(Train_X_Tfidf))

    df_tfidf.to_csv('Data Hasil TFIDF.csv')

    k=int(nilai_k)
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')


    clf = knn.fit(Train_X_Tfidf, y_train)
    predicted = clf.predict(Test_X_Tfidf)



    fileFinal = open("static/files/Data Hasil Pengujian.csv", 'w', newline='', encoding='utf-8')
    writerFinal = csv.writer(fileFinal)
    for i in range(len(predicted)):
        tweet_data = X_test.tolist()[i]
        label_data = y_test.tolist()[i]
        prediksi_data = predicted.tolist()[i]
        
        data = [tweet_data, label_data, prediksi_data ]
        writerFinal.writerow(data)
    # menyimpan tfidf
    df_tfidf = pd.DataFrame(data=csr_matrix.todense(Train_X_Tfidf))
    df_tfidf.index = X_train
    df_tfidf.columns = names
    df_tfidf.to_csv('static/files/Data Hasil TFIDF.csv')

  
    

    # menyimpan hasil klasifikasi knn ke csv
    report = classification_report(y_test,predicted, output_dict=True)
    df_knn = pd.DataFrame(report).transpose()
    df_knn.to_csv('static/files/Data Hasil Klasifikasi.csv', index= True)
    # menyimpan confusion matrix ke csv
    # confusion matrix
    unique_label = np.unique([y_test, predicted])
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, predicted, labels=unique_label), 
        index=['{:}'.format(x) for x in unique_label], 
        columns=['{:}'.format(x) for x in unique_label]
    )    
    cmtx.to_csv('static/files/Data Hasil Confusion Matrix.csv', index= True)
    # buka file
    df = pd.read_csv('static/files/Data Hasil Confusion Matrix.csv', sep=",")
    
    df.rename( columns={'Unnamed: 0':''}, inplace=True )

    df2 = pd.read_csv('static/files/Data Hasil Klasifikasi.csv', sep=",")
    df2.rename( columns={'Unnamed: 0':''}, inplace=True )

    akurasi = round(accuracy_score(y_test, predicted)  * 100, 2)
    

    flash('Berhasil Klasifikasi data ', 'klasifikasi_category')




def visualisasi_data():
    global labels, sizes
    # buka csv
    data = pd.read_csv("static/files/Data Hasil Labelling.csv")
    tweet = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    # membuat visualisasi wordcloud
    kalimat = ""

    for i in tweet.tolist():
        s =("".join(i))
        kalimat += s
    
   

    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/belajar-ngoding-codepolitan.appspot.com/o/gojek.jpg?alt=media&token=98cff15e-d00d-4330-bf95-3017b43d609b", 'gojek.jpg')
    mask = np.array(Image.open("gojek.jpg"))
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color=None, mode='RGBA', mask=mask)
    wordcloud.generate(kalimat)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('static/files/wordcloud.png', transparent=True)

    
    numbers_list = y.tolist()
    counter = dict((i, numbers_list.count(i)) for i in numbers_list)
    isPositive = 'Positif' in counter.keys()
    isNeutral = 'Netral' in counter.keys()
    isNegative = 'Negatif' in counter.keys()
    

    positif = counter["Positif"] if isPositive == True  else 0
    netral = counter["Netral"] if isNeutral == True  else 0
    negatif = counter["Negatif"] if isNegative == True  else 0
    

    sizes = [positif, netral, negatif]
    labels = ['Positif', 'Netral', 'Negatif']
@app.route("/")
def index():
    return render_template('index.html')


@app.route("/crawling",  methods=['POST', 'GET'])
def crawling():
    if request.method == 'POST':
        query = request.form.get('query')
        jumlah = request.form.get('jumlah')

        if request.form.get('crawling') == 'Crawling Data':
            crawling_data(query, jumlah)
            return render_template('crawling.html', value=analysis_result)
    return render_template("crawling.html")



ALLOWED_EXTENSION = set(['csv'])

def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():

    if request.method == 'POST':

        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if 'file' not in request.files:
                flash('Tidak Ada File', 'upload_category')
                return render_template('preprocessing.html', value=preprocessing_result)
            if not allowed_files(file.filename):
                flash('Format salah ', 'upload_category')
                return render_template('preprocessing.html', value=preprocessing_result)
            if file and allowed_files(file.filename):
                flash('Berhasil', 'upload_category')
                file.save("static/files/Data Mentah Twitter.csv")
                return render_template('preprocessing.html', value=preprocessing_result)

        preprocessing_result.clear()
        if request.form.get('preprocessing') == 'Preprocessing Data':
            preprocessing_data()
            return render_template('preprocessing.html', value=preprocessing_result)

    elif request.method == 'GET':
        return render_template('preprocessing.html', value=preprocessing_result)
    return render_template("preprocessing.html")


            



@app.route("/labelling",  methods=['POST', 'GET'])
def labelling():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if 'file' not in request.files:
                flash('Tidak Ada File', 'upload_category')
                return render_template('labelling.html', value=labelling_result)
            if not allowed_files(file.filename):
                flash('Format salah ', 'upload_category')
                return render_template('labelling.html', value=labelling_result)
            if file and allowed_files(file.filename):
                flash('Berhasil', 'upload_category')
                file.save("static/files/Data Hasil Preprocessing.csv")
                return render_template('labelling.html', value=labelling_result)

        labelling_result.clear()
        if request.form.get('labelling') == 'Labelling Data':
            labelling_data()
            return render_template('labelling.html', value=labelling_result)
    return render_template("labelling.html")


@app.route("/klasifikasi",  methods=['POST', 'GET'])
def klasifikasi():
    if request.method == 'POST':
        if request.form.get('upload') == 'Upload Data':
            file = request.files['file']
            if 'file' not in request.files:
                flash('Tidak Ada File', 'upload_category')
                return render_template('klasifikasi.html')
            if not allowed_files(file.filename):
                flash('Format salah ', 'upload_category')
                return render_template('klasifikasi.html')
            if file and allowed_files(file.filename):
                flash('Berhasil', 'upload_category')
                file.save("static/files/Data Hasil Labelling.csv")
                return render_template('klasifikasi.html')

        if request.form.get('klasifikasi') == 'Klasifikasi Data':
            nilai = request.form.get('nilai_k')
            klasifikasi_data(nilai)
            return render_template('klasifikasi.html', accuracy=akurasi, tables=[df.to_html(classes='table table-bordered table-striped fs--1 mb-0', index=False, justify='left')], titles=df.columns.values, tablesClassification=[df2.to_html(classes='table table-bordered table-striped fs--1 mb-0', index=False, justify='left')], titles2=df2.columns.values)
        if request.form.get('visualisasi') == 'Visualisasi Data':
            visualisasi_data()
            return redirect(url_for('visualisasi'))
    
    return render_template("klasifikasi.html")

@app.route("/visualisasi")
def visualisasi():
    return render_template('visualisasi.html', labels=labels, values=sizes)




if __name__ == '__main__':
    app.run(debug=True)
