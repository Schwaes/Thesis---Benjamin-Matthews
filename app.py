import customtkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import json
from ARIMA import ARIMA_predict
from ARDL import ARDL_predict

dates = np.load('dates.npy', allow_pickle=True)

with open('SalesPlotData.json') as fp:
	sales_data = json.load(fp)

# Takes item SB code as input, returns sales data in tuple (dates, sales)
def getVals(item):
	x = dates.tolist()
	x.reverse()
	y = sales_data[item]
	y.reverse()
	return (x, y)

# Class to plot all data in the app, uses pyplot
class plot:
	def __init__(self, window, item):
		self.item = item
		self.window = window
		self.x, self.y = getVals(self.item)
		_, self.arimapredy = ARIMA_predict(self.item)
		self.actual, self.ardlpredy = ARDL_predict(self.item)

	#drawing the plot and packing it into tkinter 
	def draw(self):
		self.fig = Figure(figsize=(12,6))
		a = self.fig.add_subplot(111)
		a.plot(self.x, self.y, label = 'Actual Sales')
		a.plot(self.arimapredy, ':', label = 'ARIMA prediction')
		a.plot(self.ardlpredy, '--', label = 'ARDL prediction')
		a.set_title (f"ARDL prediction for next timestep: {self.actual} sales", fontsize=12)
		self.fig.suptitle(f"Sales History for item: {self.item}", fontsize=16)
		a.set_ylabel("Units Sold", fontsize=12)
		a.set_xlabel("Date (Year Month)", fontsize=12)
		a.legend()
		a.xaxis.set_major_locator(plt.MaxNLocator(5))
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
		self.widget = self.canvas.get_tk_widget()
		self.widget.pack(pady = 40, padx = 20, expand = True)

	#removing plot to make space for another one
	def delete(self):
		self.widget.destroy()

#removes and then creates the currently displayed plot
def createPlot(frame, sku):
	global currentPlot
	try:
		currentPlot.delete()
	except:
		pass
	currentPlot = plot(frame, sku)
	currentPlot.draw()

#APP main - customtkinter app
def main():

	tk.set_appearance_mode('light')
	tk.set_default_color_theme('blue')

	root = tk.CTk()
	root.geometry('1400x700')
	root.title('Sales Forecasting Tool')

	frame = tk.CTkFrame(master=root)
	frame.pack(pady = 20, padx = 20, fill = 'both', expand = True)

	def enter(event):
		sku = event.widget.get()
		createPlot(frame, sku)

	itemBox = tk.CTkEntry(master = frame, placeholder_text = 'Item SKU')
	itemBox.bind("<Return>", enter)
	itemBox.pack(pady = 20, padx = 20)

	buttonPlot = tk.CTkButton(master = frame, command = lambda: createPlot(frame, itemBox.get()), text = 'Plot Item Sales')
	buttonPlot.pack()

	root.mainloop()



main()
