from sinogram import *
from reversesinogram import *
from dicom import *
from ipywidgets import interact, widgets
import cv2
from IPython.display import display
from tkinter import *
from tkinter.filedialog import askopenfilename

def choose_file(window_title):
    root = Tk()
    filename = askopenfilename(initialdir = "/",title = window_title,filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    root.destroy()
    return filename

def display_texts():
    print("Wypełnij informacje do DICOM'a")
    display(text_name)
    display(text_image_comment)
    display(text_study_date)
    display(text_filename)



    

def show_image():
    plt.imshow(im, cmap='gray')
    
def display_sliders():  
    print("Ustaw parametry sinogramu")
    display(slider_l,"Ustaw wartość L")
    display(slider_alpha,"Ustaw wartość alpha")
    display(slider_detectors,"Ustaw ilość detektorów")
    display(checkbox_filtr)
    

def save_dicom_data(image): 
    save_dicom(text_name.value,text_image_comment.value,
                                  text_study_date.value,image,text_filename.value) 

def load_dicom_data():
    load_dicom(text_filename.value)
    
def make_sinogram():
    return sinogram(slider_alpha.value,slider_detectors.value,R, slider_l.value, im, checkbox_filtr.value)
    
def make_reverse_sinogram(Sin):
    return reverseSinogram(slider_alpha.value,slider_detectors.value,R,R_org, slider_l.value, Sin,im)

def sinograms():
    Sin = make_sinogram()
    Gif = make_reverse_sinogram(Sin)
    plt.subplots(figsize=(20, 10))
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(im_org, cmap='gray')
    plt.title('Obraz oryginalny')
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(Sin, cmap='gray')
    plt.title('Sinogram')
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(Gif[-1], cmap='gray')
    plt.title('Efekt końcowy')
    return Gif

def DICOMs():
    save_dicom_data()
    load_dicom_data()
    
    
slider_l = widgets.IntSlider(min=45,max=270,step=45,value=180)
slider_alpha = widgets.FloatSlider(min=0,max=4,step=0.1,value=2.0)
slider_detectors = widgets.IntSlider(min=90,max=720,step=90,value=180)
checkbox_filtr = widgets.Checkbox(
    value=False,
    description='Czy zastosowac filtr',
    disabled=False,
    indent=False
)
text_name = widgets.Text(
    value='',
    placeholder='Patient Name',
    description='',
    disabled=False
)
text_image_comment = widgets.Text(
    value='',
    placeholder='Image comment',
    description='',
    disabled=False
)
text_study_date = widgets.Text(
    value='',
    placeholder='Study date',
    description='',
    disabled=False
)
text_filename = widgets.Text(
    value='',
    placeholder='Filename',
    description='',
    disabled=False
)



filename = choose_file("Wybierz zdjecie do sinogramu")

print("Wybrano plik: ",filename)
im_org = cv2.imread(filename, 0)

def rmse(Y):
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(im_org, Y))
    return rmse

R_org = min(im_org.shape)/2

gif = []

im = cv2.copyMakeBorder(im_org, int(R_org/2), int(R_org/2), int(R_org/2), int(R_org/2),cv2.BORDER_CONSTANT,None, 0)
R = min(im.shape)/2

