from tkinter import *
import os
from tkinter import messagebox, filedialog
import imageio
from PIL import Image, ImageTk



class Application(Frame):
    def __init__(self, save_dir, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.place()
        self.current_idx = 0
        self.filedir = filedialog.askdirectory()
        self.img_paths = [os.path.join(self.filedir,img_name) for img_name in os.listdir(self.filedir)]
        self.save_dir = save_dir
        self.current_label = None
        # 记录下鼠标左键按下的坐标
        self.X = 0
        self.Y = 0

        # 要处理的目录数
        self.dirs_num = len(os.listdir(os.path.dirname(self.filedir)))
        
        self.photo1 = None  # 操作画布的全局photo
        self.photo2 = None  # 展示画布的全局photo

        self.rectangle = None
        self.origin_img = None
        self.rectangle_scale = 1 # 截取矩形的长宽比，=矩形长：矩形宽
        self.rectangle_width = 100 # 截取矩形的默认长
        self.rectangle_adjust = 1
        self.createWidget()
        
    def createWidget(self):
        self.txt_content = StringVar()
        self.current_label = Label(self, textvariable=self.txt_content).grid(column=0, row=0, columnspan=3)
        self.txt_content.set("当前打开的目录为："+self.filedir+"\t\t\t截取图片保存目录为："+self.save_dir)

        self.canvas = Canvas(self, width=600, height=350, bg='white')
        self.canvas.grid(column=0, row=1, rowspan=9)
        self.txt_index = StringVar()
        self.index_label = Label(self, textvariable=self.txt_index).grid(column=0, row=10)
        self.txt_index.set('当前图片名：'+os.path.basename(self.img_paths[self.current_idx]))

        self.btn01 = Button(self, text='打开', command=self.get_new_dir, bg='white', anchor='s')
        self.btn01.grid(column=1, row=1)
        self.btn02 = Button(self, text='重画', command=self.clear_rectangle, bg='white', anchor='s')
        self.btn02.grid(column=1, row=2)
        self.btn03 = Button(self, text='截取', command=self.crop, bg='white', anchor='s')
        self.btn03.grid(column=1, row=3)
        self.btn03 = Button(self, text='上一张图', command=self.get_front, bg='white', anchor='s')
        self.btn03.grid(column=1, row=4)
        self.btn03 = Button(self, text='下一张图', command=self.get_next, bg='white', anchor='s')
        self.btn03.grid(column=1, row=5)

        def scale_crop(V):
            if self.rectangle is not None:
                self.clear_rectangle()
                self.rectangle_adjust = float(V)
                self.rectangle = self.canvas.create_rectangle(self.X, self.Y, (self.X+self.rectangle_width)*self.rectangle_adjust, (self.Y+self.rectangle_width//self.rectangle_scale)*self.rectangle_adjust, 
                                outline='red', width=3)
                    
        def hw_scale_crop(V):
            if self.rectangle is not None:
                self.clear_rectangle()
                self.rectangle_scale = float(V)
                self.rectangle = self.canvas.create_rectangle(self.X, self.Y, (self.X+self.rectangle_width)*self.rectangle_adjust, (self.Y+self.rectangle_width//self.rectangle_scale)*self.rectangle_adjust, 
                                outline='red', width=3)

        self.scale_b = Scale(self, orient='vertical', label='区域放缩', from_=1, to=2, resolution=0.05, command=scale_crop)
        self.scale_b.grid(column=1, row=6)

        self.scale_b = Scale(self, orient='vertical', label='长宽比例', from_=1, to=3, resolution=0.05, command=hw_scale_crop)
        self.scale_b.grid(column=1, row=7)

        self.show_crop = Canvas(self, width=600, height=350, bg='white')    # 用于显示裁剪结果的画布
        self.show_crop.grid(column=2, row=1, rowspan=10)

        self.btn03 = Button(self, text='上一文件夹', command=self.get_front_dir, bg='white', anchor='s')
        self.btn03.grid(column=1, row=8)
        self.btn03 = Button(self, text='下一文件夹', command=self.get_next_dir, bg='white', anchor='s')
        self.btn03.grid(column=1, row=9)

        self.origin_img = Image.open(self.img_paths[self.current_idx])
        self.photo1 = ImageTk.PhotoImage(self.origin_img)
        self.canvas.create_image((300, 175), image=self.photo1)

        def onLeftButtonDown(event):
            self.clear_rectangle()
            self.X = event.x
            self.Y = event.y
            self.rectangle = self.canvas.create_rectangle(self.X, self.Y, (self.X+self.rectangle_width)*self.rectangle_adjust, (self.Y+self.rectangle_width//self.rectangle_scale)*self.rectangle_adjust, 
                            outline='red', width=3)

        self.canvas.bind('<Button-1>', onLeftButtonDown)
    
    def clear_rectangle(self):
        if self.rectangle is not None:
            self.canvas.delete(self.rectangle)
            self.rectangle = None
            self.photo2 = None

    def get_next_dir(self):
        self.current_idx = 0
        current_dir_num = int(os.path.basename(self.filedir))
        if current_dir_num >= self.dirs_num:
            messagebox.showinfo(title='消息提示框', message='所有文件夹都处理完了！')
        else:
            self.filedir = os.path.join(os.path.dirname(self.filedir),str(current_dir_num+1))
            self.img_paths = [os.path.join(self.filedir,img_name) for img_name in os.listdir(self.filedir)]
            self.origin_img = Image.open(self.img_paths[self.current_idx])
            self.photo1 = ImageTk.PhotoImage(self.origin_img)
            self.canvas.create_image((300, 175), image=self.photo1)
            self.txt_content.set("当前打开的目录为："+self.filedir+"\t\t\t截取图片保存目录为："+self.save_dir)
            self.txt_index.set('当前图片名：'+os.path.basename(self.img_paths[self.current_idx]))

    def get_front_dir(self):
        self.current_idx = 0
        current_dir_num = int(os.path.basename(self.filedir))
        if current_dir_num <= 1:
            messagebox.showinfo(title='消息提示框', message='现在是第1个文件夹！')
        else:
            self.filedir = os.path.join(os.path.dirname(self.filedir),str(current_dir_num-1))
            self.img_paths = [os.path.join(self.filedir,img_name) for img_name in os.listdir(self.filedir)]
            self.origin_img = Image.open(self.img_paths[self.current_idx])
            self.photo1 = ImageTk.PhotoImage(self.origin_img)
            self.canvas.create_image((300, 175), image=self.photo1)
            self.txt_content.set("当前打开的目录为："+self.filedir+"\t\t\t截取图片保存目录为："+self.save_dir)
            self.txt_index.set('当前图片名：'+os.path.basename(self.img_paths[self.current_idx]))

    def get_next(self):
        self.photo2 = None
        if self.current_idx >= len(self.img_paths)-1:
            messagebox.showinfo(title='消息提示框', message='文件夹没有其他图片了！')
        else:
            self.current_idx += 1
            self.origin_img = Image.open(self.img_paths[self.current_idx])
            self.photo1 = ImageTk.PhotoImage(self.origin_img)  # 实际上是把这个变成全局变量
            self.canvas.create_image((300, 175), image=self.photo1)
            self.txt_index.set('当前图片名：'+os.path.basename(self.img_paths[self.current_idx]))
    
    def get_front(self):
        self.photo2 = None
        if self.current_idx == 0:
            messagebox.showinfo(title='消息提示框', message='现在是第一张图片！')
        else:
            self.current_idx -= 1
            self.origin_img = Image.open(self.img_paths[self.current_idx])
            self.photo1 = ImageTk.PhotoImage(self.origin_img)  # 实际上是把这个变成全局变量
            self.canvas.create_image((300, 175), image=self.photo1)
            self.txt_index.set('当前图片名：'+os.path.basename(self.img_paths[self.current_idx]))

    def crop(self):
        if self.rectangle is not None:
            self.croped_img = self.origin_img.crop((self.X, self.Y, self.X+self.rectangle_width, self.Y+self.rectangle_width//self.rectangle_scale))
            self.photo2 = ImageTk.PhotoImage(self.croped_img.resize((600,350)))
            self.show_crop.create_image((300, 175), image=self.photo2)
            # 保存截取的图片
            base_dir = os.path.join(self.save_dir,os.path.basename(self.filedir))
            if os.path.exists(base_dir) is False:
                os.makedirs(base_dir)
            self.croped_img.save(os.path.join(base_dir,str(self.current_idx)+'.png'))
            
    def get_new_dir(self):
        self.current_idx = 0
        self.filedir = filedialog.askdirectory()
        self.img_paths = [os.path.join(self.filedir,img_name) for img_name in os.listdir(self.filedir)]
        self.origin_img = Image.open(self.img_paths[self.current_idx])
        self.photo1 = ImageTk.PhotoImage(self.origin_img)
        self.canvas.create_image((300, 175), image=self.photo1)
        self.txt_content.set("当前打开的目录为："+self.filedir+"\t\t\t截取图片保存目录为："+self.save_dir)

root = Tk()
root.geometry('1400x430')

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir,'croped_images')
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

app = Application(save_path, master=root)

root.mainloop()
