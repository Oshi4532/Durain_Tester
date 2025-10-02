import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
import datetime
from tkinter import filedialog
from ultralytics import YOLO


class Durain_Tester:
    def __init__(self, root):
        self.root = root
        self.root.title("Durain_Tester")
        self.root.geometry("800x600")
        self.root.config(bg = "#324704")
        self.showing = False
        self.root.iconphoto(False, ImageTk.PhotoImage(file = 'icon.webp') )

        image_frame = tk.Frame(root, bg="#324704")
        image_frame.pack(expand=True)

        self.label = Label(image_frame, bg="#324704")
        self.label.pack(expand=True)

        button_frame = tk.Frame(root, bg="#324704")
        button_frame.pack(expand=True)





        AI_model = YOLO('darach.pt')
        self.model = AI_model 
        

        
        self.label.pack()
        
        #import
        self.import_button = Button(button_frame, text="Import", command=self.import_pic)
        self.import_button.pack(side=tk.LEFT, padx=10)
        #clear
        self.clear_button = Button(button_frame, text="Clear", command=self.clear_image)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        button_frame.pack(anchor="center")




    def import_pic(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image File","*.jpg;*.jpeg;*.png;*.bmp;*.gif")])

        if file_path:
            image_bgr = cv2.imread(file_path)
            image_resized = cv2.resize(image_bgr, (640, 480))
            results = self.model.predict(source=image_resized, imgsz=640, conf=0.25, verbose=False)
            result = results[0]



            class_ids  = result.boxes.cls.cpu().numpy() if result.boxes else []
            num_NSCLC = 0
            num_SCLC = 0

            for class_id in class_ids:
                if int(class_id) == 0:
                    num_NSCLC += 1 
                elif int(class_id) == 1:
                    num_SCLC += 1


            detected_frame = results[0].plot()

            detected_RGB = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            self.current_frame = Image.fromarray(detected_RGB)  
            self.photo = ImageTk.PhotoImage(image=self.current_frame)
            
            self.label.config(image=self.photo) 
            self.showing = True
    
    def clear_image(self):
        self.label.config(image="")
        self.showing = False

    

        


if __name__ == "__main__":
    root = tk.Tk()
    app = Durain_Tester(root)
    root.mainloop()