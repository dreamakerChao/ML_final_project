import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image,ImageTk
from keras.models import load_model
import numpy as np
import copy
import pandas as pd
classifacation=pd.read_csv('class.csv')
classifacation=np.array(classifacation)
img_matrix =[]

def show_import_img():
    file_path = filedialog.askopenfilename()
    if not file_path:
        messagebox.showinfo("Info", "Import failed")
    else:
        path_list=file_path.split('/')
        path=path_list[len(path_list)-3]+'/'+path_list[len(path_list)-2]+'/'+path_list[len(path_list)-1]
        lb_input.configure(text ='path : '+path,fg='white')
        import_img(file_path)
    

def import_img(path):
    try:
        img = Image.open(path)
        tk_img = ImageTk.PhotoImage(img)
        w, h = img.size
        if w<500 or h<500:
            pil_image_resized=img.resize((500,500))
        else:
            pil_image_resized=img
        tk_img = ImageTk.PhotoImage(pil_image_resized)
        canvas.delete('all')
        canvas.config(scrollregion=(0,0,w,h))
        canvas.create_image(0, 0, anchor='nw', image=tk_img)
        canvas.tk_img = tk_img
        global img_matrix      
        img=img.convert('RGB')
        img=img.resize((128,128))
        img= np.array(img)
        img_matrix.append(img)
        img_matrix=np.array(img_matrix)
        #img_matrix=img_matrix.reshape((1,64,64,3))
        print(img_matrix)
        image2 = Image.fromarray(img.reshape(128,128,3))
        
        image2.show()
        print(img_matrix.shape)
        messagebox.showinfo("Info", "Import successfully")
    except:
        messagebox.showinfo("Info", "Import failed")

def predict_model():
    model = load_model('410987011_model.h5')
    #model.summary()
    output=model.predict(img_matrix)
    output=np.ravel(output)
    i=0
    for item in output:
        if item !=0:
            break
        i+=1
    
    lb_output.configure(text =f'Output : {classifacation[i]}',fg='white',font=("Arial",14))

#window
window = tk.Tk()
window.title('Traffic sign recognize By ML_410987011')
window.geometry('600x600')
window.resizable(False,True)

#canvas
canvas = tk.Canvas(window, width=500, height=500, bg='#6C6C6C')
canvas.create_text(250, 250, text='NO file input yet!',fill='white')
#canvas.pack(padx='50')
#canvas scollbar
scrollX = tk.Scrollbar(window, orient='horizontal')
scrollX.pack(side='bottom', fill='x')
scrollX.config(command=canvas.xview)

scrollY = tk.Scrollbar(window, orient='vertical')
scrollY.pack(side='right', fill='y')
scrollY.config(command=canvas.yview)
canvas.config(xscrollcommand=scrollX.set, yscrollcommand=scrollY.set)
canvas.pack(side='left')
#button
bt_import_image=tk.Button(text='import image',width=11,command=show_import_img,activebackground='Green',activeforeground='white')
bt_predict=tk.Button(text='Predict', width=11,activebackground='Green',activeforeground='white',command=predict_model)
#label
lb_input=tk.Label(anchor='nw',text='input :  NONE',width=50,bg='gray',fg='white')
lb_output=tk.Label(anchor='nw',text='output :  ',width=60,height=15,bg='gray',fg='white')
#layout
bt_import_image.place(x=20, y=10)
bt_predict.place(x=120, y=10)
lb_input.place(anchor='nw',x=220,y=10)
lb_output.place(anchor='nw',x=0,y=560)
#mainloop

window.mainloop()