from tkinter import *
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from pyscreenshot import grab
import tensorflow as tf
gui = Tk()
gui.geometry("735x445+250+170")
gui.title("Digit Detector")
gui.iconbitmap("D:\\CODES\\MiniProject\\DigitRecogination\\Digit_icon.ico")

bg = PhotoImage(file = "C:\\Users\\Akanksha Uniyal\\Downloads\\Digit1.png")


model = tf.keras.models.load_model('DigitRecogModel.h5')
canvas1 = Canvas( gui, width = 50,height = 180)
canvas1.create_image( 0, 0, image = bg ,anchor = "nw")
canvas1.pack(fill = "both", expand = True)
#model = tf.keras.models.load_model('temp1.h5')
canvas = Canvas(canvas1, width=300, height=300, bg = "black", cursor="cross")

global d
global text_box
def predict(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #img.show()
    #predicting the class
    res = model.predict([img])
    return np.argmax(res) 
    
def classify():
    im = grab(bbox=(360,330, 740, 710))
    #im.show()
    digit= predict(im)
    d= str(digit)
    print("digit is -- ",digit)
    text_box.configure(state='normal')
    text_box.insert(INSERT,d)
    text_box.configure(state='disabled')
    
    #label1 = Label(gui, text=d, font = "lucida 150")
    #
    
def clear_all():
        canvas.delete("all")
        text_box.configure(state='normal')
        text_box.delete("1.0", END);
        text_box.configure(state='disabled')

def draw_lines(event):
        x = event.x
        y = event.y
        r=8
        canvas.create_oval(x-r, y-r, x + r, y + r, fill="white",outline= "white")

label1 = Label(canvas1, text="   Prediction", font = "lucida 20",bg = "white")
text_box = Text(canvas1, height = 1, width = 1, font = "lucida 170",borderwidth=4)
label = Label(canvas1, text="Digit Recognition using Convolution Neural Network ", font=20,bg = "white")
classify_btn = Button(canvas1, text = "Recognise", command = classify,font=8,borderwidth=8) 
button_clear = Button(canvas1, text = "Clear", command = clear_all,font=8,borderwidth=8)

# Grid structure

canvas.grid(row=1, column=1, pady=2, padx = 30,sticky=W )
label.grid(row=0, column=1,pady=15, padx=35)
classify_btn.grid(row=2, column=2, pady=10, padx=10)
button_clear.grid(row=2, column=1,pady=10, padx=150,sticky=W)
text_box.grid(row=1, column=2,pady=15, padx=2)
label1.grid(row=0, column=2,pady=15, padx=40)
canvas.bind("<B1-Motion>", draw_lines)


gui.mainloop()
