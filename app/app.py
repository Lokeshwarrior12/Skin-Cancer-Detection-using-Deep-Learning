# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup,Response, render_template_string,redirect,url_for
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model, Sequential
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openai
import json
import webbrowser
# import Image
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image

# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==============================================================================================

model_path="EfficientNetB3-skindisease-83.00.h5"
model=load_model(model_path)


def predictor(sdir, csv_path,  crop_image = False):    
    # read in the csv file
    class_df=pd.read_csv(csv_path,encoding='cp1252')    
    # img_height=int(class_df['height'].iloc[0])
    img_height=int(class_df['width'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)
    scale=1
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1]) 
        print (s1,s2)
    path_list=[]
    paths=sdir
    # print('path',paths)
    # for f in paths:
    path_list.append(paths)
    image_count=1    
    index_list=[] 
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):
               
        img=cv2.imread(path_list[i])
        # print('i',img)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_image == True:
            status, img=crop(img)
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)            
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)
            p= np.squeeze (model.predict(img))           
            index=np.argmax(p)            
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count==1:
        # print(class_df.columns.tolist())
        class_name= class_df['class'].iloc[index_list[0]]
        symtom=class_df['symtoms '].iloc[index_list[0]]
        # symtom='1'
        medicine=class_df['medicine'].iloc[index_list[0]]
        wht=class_df['what is'].iloc[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0] 
        # plt.title(class_name, color='blue', fontsize=16)
        # plt.axis('off')
        # plt.imshow(img)
        # print(symtom,medicine,wht)
        return class_name, probability,symtom,medicine,wht
    elif good_image_count == 0:
        return None, None,None,None,None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]            
            if nkey == key:
                keycount +=1                
        if keycount> most:
            most=keycount
            isave=i             
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    class_name=class_df['class'].iloc[best_index]
    symtom=class_df['symtoms '].iloc[best_index]
    medicine=class_df['medicine'].iloc[best_index]
    wht=class_df['what is'].iloc[best_index]
    # print(symtom,medicine,wht)
    # plt.title(class_name, color='blue', fontsize=16)
    # plt.axis('off')
    # plt.imshow(img)
    return class_name, bestsum/image_count,symtom,medicine,wht
img_size=(300, 300)

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
openai.api_key = "your_api_key"

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


import base64
def render_picture(data):

    render_pic = base64.b64encode(data).decode('ascii') 
    return render_pic

@app.route('/disease')
def disease():
    return render_template('disease.html')


@app.route('/disease-predict', methods=['POST','GET' ])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method:
        if request.method == 'POST':
            img1 = request.files['file1']
           
                
        else:
            img1 = request.args.get('file1')
         
        img1.save("out.jpg")   
              
        path="out.jpg"
        
        csv_path="class.csv"
        model_path="EfficientNetB3-skindisease-83.00.h5"
        class_name, probability,symtom,medicine,wht=predictor(path, csv_path, crop_image = False)
       
        print(symtom)

        prediction = Markup(class_name)
            
        
        return render_template('disease-result.html', prediction=prediction,symtom=symtom,medicine=medicine,wht=wht, title=title,prob=probability)


@app.route('/live1')
def live1():
    title = 'Harvestify - Disease Detection'
    path = "out.jpg"
    csv_path = "class.csv"

    
    class_name, probability, symtom, medicine, wht = predictor(path, csv_path, crop_image=False)
    prediction = Markup(class_name)
    print(probability)

    # print(symtom,medicine,wht)
    # print(symtom)

    return render_template('disease-result.html', prediction=prediction, symtom=symtom, medicine=medicine, wht=wht, title=title, prob=probability)


global capture, out
capture = 0
switch = 1

try:
    os.mkdir('./shots')
except OSError as error:
    pass

camera = cv2.VideoCapture(0)

def gen_frames():
    import datetime
    global capture, out
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
                path = "out.jpg"
                cv2.imwrite(path, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass
  

@app.route('/live')
def live():
    return render_template('in.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        message = request.form['message']
        return 'Message sent!'
    else:
        return render_template('contact.html')
    

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/doctor')
def doctor():
    return render_template('doctor.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/blog-details')
def blogdetails():
    return render_template('blog-details.html')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            switch=0
            camera.release()
            cv2.destroyAllWindows()
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('predict') == 'predict':
            switch=0
            camera.release()
            cv2.destroyAllWindows()
            return redirect(url_for('live1'))
                          
                 
    elif request.method=='GET':
        return render_template('in.html')
    return render_template('in.html')


# create a database connection
conn = sqlite3.connect('database.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (email text PRIMARY KEY, name text, password text)''')
c.execute('''CREATE TABLE IF NOT EXISTS appointments
             (name text, date text, details text)''')
conn.commit()

# close the connection
conn.close()


@app.route('/login', methods=['GET', 'POST'])
def login():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = c.fetchone()

        if user:
            # Successful login
            conn.close()
            return redirect(url_for('home'))
        else:
            # Failed login
            conn.close()
            return render_template('login.html', message='Invalid email or password')

    # GET request
    conn.close()
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']

        c.execute("INSERT INTO users (email, name, password) VALUES (?, ?, ?)", (email, name, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    conn.close()
    return render_template('login.html')


@app.route('/send_email', methods=['POST'])
def send_email():
    email = request.form['email']
    name = request.form['name']
    date = request.form['date']
    time = request.form['time']
    phone = request.form['number']
    details = request.form['details']

    message = f"Hi,\n\n{name} has requested an appointment on {date} at {time}.\n Please conform your Phone number{phone} and \nDetails:\n{details}"

    sender_email = 'your_email'
    sender_password = 'your_password'
    recipient_email = email

    try:
        # Set up the SMTP server and login
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # Compose the email message
        msg = MIMEText(message)
        msg['Subject'] = 'Appointment Request'
        msg['From'] = sender_email
        msg['To'] = recipient_email

        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()

        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS appointments
                     (name text, date text, details text)''')
        c.execute('''INSERT INTO appointments (name, date, details)
                     VALUES (?, ?, ?)''', (name, date, details))
        conn.commit()
        conn.close()

        return 'Appointment request sent successfully.\n Please Check your inbox'
    except Exception as e:
        return f'An error occurred while sending the email: {str(e)}'

    

# Set up the OpenAI GPT-3 API parameters
model_engine = "text-davinci-002"
temperature = 0.5
max_tokens = 50
top_p = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0

# Define a function to generate text using the OpenAI API
def generate_text(prompt_text):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt_text,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response.choices[0].text.strip()

@app.route('/ChatGPT')
# Define a route to render the ChatGPT page
def chat():
    prompt = "Hello, how can I help you today?"
    generated_text = generate_text(prompt)
    return render_template('ChatGPT.html', generated_text=generated_text)

# Define the chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        # Get the message from the user
        message = request.form.get('message')

        if not message:
            # Render the chatbot HTML template with an error message
            error_message = "Please enter a message"
            return render_template('ChatGPT.html', error_message=error_message)

        try:
            # Use OpenAI to generate a response
            response = openai.Completion.create(
                engine="davinci",
                prompt=message,
                max_tokens=60,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Extract the response from OpenAI's JSON response
            response_text = response.choices[0].text.strip()

            # Render the chatbot HTML template with the response
            return render_template('ChatGPT.html', response=response_text)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            return render_template('ChatGPT.html', error_message=error_message)

    else:
        return render_template('ChatGPT.html')


import webbrowser

@app.route('/google', methods=['GET', 'POST'])
def google():
    if request.method == 'POST':
        
        search_query = request.form['search_query']

        google_url = f"https://www.google.com/search?q={search_query}"

        webbrowser.open_new_tab(google_url)

        return "Google search successful."
    else:
        return render_template('ChatGPT.html')


@app.route('/google1', methods=['GET', 'POST'])
def google1():
    if request.method == 'POST':
        search_query1 = request.form['search_query']

        google_url1 = f"https://www.google.com/search?q={search_query1}"

        webbrowser.open_new_tab(google_url1)

        return "Google search successful."
    else:
        return render_template('ChatGPT.html')



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)

camera.release()
cv2.destroyAllWindows()
