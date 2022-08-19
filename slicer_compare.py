"""
przykład okienka wyświetlanai danych 3d przez przekroje
"""

import time
from typing import List

import PyQt5.QtWidgets as QtGui
from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
from IPython import get_ipython
from PyQt5.QtGui import QTransform

get_ipython().magic("gui qt5")


class Adjuster:
    pass

class SpinBox(QtGui.QDoubleSpinBox):
    def __init__(self, widget):
        super().__init__(widget)
        self.scale = 1
        self.transpose = 0

    def textFromValue(self, value):
        text = str(value*self.scale+self.transpose)
        return text.replace('.', ',')

    def valueFromText(self, text):
        text = text.replace(',', '.')
        return int((float(text) - self.transpose) / self.scale)


class Slider(Adjuster):
    scale = 1
    def __init__(self, v0, min, max):
        #self.label = label
        self.v0, self.min, self.max = v0, min, max
        self.onUpdate = lambda : None
        self.init_widget()


    def init_widget(self):
        self.widget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self.widget)
        #self.num = QtGui.QSpinBox(self.widget)
        self.num = SpinBox(self.widget)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.num)
        self.widget.setLayout(self.layout)

        self.slider.setRange(self.min, self.max)
        self.slider.setValue(self.v0)
        self.num.setRange(self.min, self.max)
        self.num.setValue(self.v0)
        self.slider.valueChanged.connect(self.upd)
        self.num.valueChanged.connect(self.upd)

    def upd(self, v):
        if self.slider.value() != int(v):
            return self.slider.setValue(int(v))
        elif self.num.value() != int(v):
            return self.num.setValue(int(v))
        else:
            #assert v<=self.max, f"error {v} > {self.max}[0]"
            self.onUpdate()

    def setRange(self, min,max):
        self.min = min
        self.max = max
        self.slider.setRange(self.min, self.max)
        self.num.setRange(self.min, self.max)

    @property
    def value(self):
        return int(self.num.value()) * self.scale


class SliceDisplay(QtGui.QMainWindow):
    status = QtCore.Signal(object)
    newData = QtCore.Signal(object)
    def __init__(self):
        self.app = pg.QtGui.QApplication.instance()
        assert self.app, "run %gui qt5 prior to loading display"
        super(self.__class__, self).__init__()
        self.cw = QtGui.QWidget(self)
        self.setCentralWidget(self.cw)
        self.status.connect(lambda s: self.statusBar().showMessage(s))
        self.l0 = QtGui.QGridLayout()
        self.cw.setLayout(self.l0)
        self.plot2d1 = pg.PlotWidget()
        self.plot2d1_viewbox = self.plot2d1.plotItem.getViewBox()
        self.l0.addWidget(self.plot2d1, 0, 0, 4, 4)
        self.plot1d1 = pg.PlotWidget()
        self.plot1d1_viewbox = self.plot1d1.plotItem.getViewBox()
        self.l0.addWidget(self.plot1d1, 5, 0, 4, 4)
        self.plot2d2 = pg.PlotWidget()
        self.plot2d2_viewbox = self.plot2d2.plotItem.getViewBox()
        self.l0.addWidget(self.plot2d2, 0, 4, 4, 4)
        self.plot1d2 = pg.PlotWidget()
        self.plot1d2_viewbox = self.plot1d2.plotItem.getViewBox()
        self.l0.addWidget(self.plot1d2, 5, 4, 4, 4)

        self.sel_field = QtGui.QComboBox(self.cw)
        self.l0.addWidget(self.sel_field, 4, 0, 1, 1)
        self.sel_field.currentIndexChanged.connect(self.changefield)


        self.curves = []
        self.sliders = {}
        self.setWindowTitle('SliceDisplay')
        self.show()
        self.activateWindow()
        self.lines = {}
        self.dim = 2

        self.histogram_lut = pg.HistogramLUTWidget()
        self.histogram_lut.item.gradient.loadPreset('flame')
        #self.l0.addWidget(self.histogram_lut, 0, 5, 3, )

    def setresult(self, r, r2):
        self.sel_field.addItems(['mod', 'phase', 'Re', 'Im'])
        data_name = self.sel_field.currentText()
        self.data1_abs = np.abs(r)
        self.data1_phase = np.unwrap(np.angle(r))
        self.data1_re = np.real(r)
        self.data1_im = np.imag(r)
        self.data2_abs = np.abs(r2)
        self.data2_phase = np.unwrap(np.angle(r2))
        self.data2_re = np.real(r2)
        self.data2_im = np.imag(r2)
        self.plot = lambda : self.plot_mod_phase()
        self.init_sequence()
        self.plot()

    def updateresult(self, r, r2):
        self.data1_abs = np.abs(r)
        self.data1_phase = np.unwrap(np.angle(r))
        self.data1_re = np.real(r)
        self.data1_im = np.imag(r)
        self.data2_abs = np.abs(r2)
        self.data2_phase = np.unwrap(np.angle(r2))
        self.data2_re = np.real(r2)
        self.data2_im = np.imag(r2)
        self.slider.setRange(0, len(self.data1_abs)-1)
        self.line1.setBounds((self.slider.min, self.slider.max))
        self.line2.setBounds((self.slider.min, self.slider.max))
        self.changefield()
        

    def init_sequence(self):
        self.slider = Slider(0, 0, len(self.data1_abs)-1)
        self.l0.addWidget(self.slider.widget, 4, 1, 1, 6)
        self.slider.onUpdate = lambda : self.plot()

        self.image2d1 = pg.ImageItem()
        self.plot2d1_viewbox.addItem(self.image2d1)
        self.histogram_lut.setImageItem(self.image2d1)
        self.image2d1.setImage(self.data1_abs)

        self.image2d2 = pg.ImageItem()
        self.plot2d2_viewbox.addItem(self.image2d2)
        self.histogram_lut.setImageItem(self.image2d2)
        self.image2d2.setImage(self.data2_abs)

        self.image1d11 = pg.PlotDataItem()
        self.plot1d1_viewbox.addItem(self.image1d11)
        self.image1d11.setData(self.data1_abs[self.slider.value])
        self.graph_update11 = lambda data : self.image1d11.setData(data)

        self.image1d12 = pg.PlotDataItem(pen='r')
        self.plot1d1_viewbox.addItem(self.image1d12)
        self.image1d11.setData(self.data2_abs[self.slider.value])
        self.graph_update12 = lambda data : self.image1d12.setData(data)

        self.image1d21 = pg.PlotDataItem()
        self.plot1d2_viewbox.addItem(self.image1d21)
        self.image1d11.setData(self.data1_phase[self.slider.value])
        self.graph_update21 = lambda data : self.image1d21.setData(data)

        self.image1d22 = pg.PlotDataItem(pen='r')
        self.plot1d2_viewbox.addItem(self.image1d22)
        self.image1d11.setData(self.data2_phase[self.slider.value])
        self.graph_update22 = lambda data : self.image1d22.setData(data)

        self.line1 = pg.InfiniteLine(pos=self.slider.value, angle=90, pen='w', movable=True, bounds = [self.slider.min, self.slider.max])
        self.plot2d1_viewbox.addItem(self.line1)
        self.line1.sigDragged.connect(lambda : self.slider.upd(self.line1.value()))

        self.line2 = pg.InfiniteLine(pos=self.slider.value, angle=90, pen='r', movable=True, bounds = [self.slider.min, self.slider.max])
        self.plot2d2_viewbox.addItem(self.line2)
        self.line2.sigDragged.connect(lambda : self.slider.upd(self.line2.value()))

    def changefield(self):
        data_name = self.sel_field.currentText()
        try:
            if data_name == 'mod':
                self.image2d1.setImage(self.data1_abs)
                self.image2d2.setImage(self.data2_abs)
                self.plot = lambda : self.plot_mod_phase()
            elif data_name == 'phase':
                self.image2d1.setImage(self.data1_phase)
                self.image2d2.setImage(self.data2_phase)
                self.plot = lambda : self.plot_mod_phase()           
            elif data_name == 'Re':
                self.image2d1.setImage(self.data1_re)
                self.image2d2.setImage(self.data2_re)
                self.plot = lambda : self.plot_re_im()  
            elif data_name == 'Im':
                self.image2d1.setImage(self.data1_im)
                self.image2d2.setImage(self.data2_im)
                self.plot = lambda : self.plot_re_im()
            self.plot()
        except AttributeError:
            pass

    def plot_re_im(self):
        try:
            self.graph_update11(self.data1_re[self.slider.value])
            self.graph_update12(self.data2_re[self.slider.value])
            self.graph_update21(self.data1_im[self.slider.value])
            self.graph_update22(self.data2_im[self.slider.value])
        except AttributeError:
            print ('No slice sequence while plotting')
        try:
            self.line1.setValue(self.slider.value)
            self.line2.setValue(self.slider.value)
        except AttributeError:
            print ('No line while plotting')
        return self

    def plot_mod_phase(self):
        try:
            self.graph_update11(self.data1_abs[self.slider.value])
            self.graph_update12(self.data2_abs[self.slider.value])
            self.graph_update21(self.data1_phase[self.slider.value])
            self.graph_update22(self.data2_phase[self.slider.value])
        except AttributeError:
            print ('No slice sequence while plotting')
        try:
            self.line1.setValue(self.slider.value)
            self.line2.setValue(self.slider.value)
        except AttributeError:
            print ('No line while plotting')
        return self