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

get_ipython().magic("gui qt5")

class Adjuster:
    pass

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
        self.num = QtGui.QSpinBox(self.widget)
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
        if self.slider.value() != v:
            return self.slider.setValue(v)
        elif self.num.value() != v:
            return self.num.setValue(v)
        else:
            #assert v<=self.max, f"error {v} > {self.max}[0]"
            self.onUpdate()

    @property
    def value(self):
        return self.num.value() * self.scale

class ConstSlice:
    def __init__(self):
        self.v = np.s_[:]

    @property
    def value(self):
        return self.v


class DimensionElement:
    def __init__(self):
        self.fixed = True
        self.sliders = []

    def wrap_param(self, l):
        if isinstance(l, (int, float)):
            return ConstSlice()
        elif isinstance(l, Slider):
            self.sliders.append(l)
            return l
        else:
            raise ValueError(l)


class Slice(DimensionElement):
    def __init__(self, l):
        super().__init__()
        self.l = self.wrap_param(l)


class SliceSequence():
    def __init__(self, *args):
        self.seq = args
        assert all(isinstance(e,DimensionElement) for e in args)
        self.sliders : List[Slider] = []
        for e in args:
            self.sliders.extend(e.sliders)

    def setdata(self, data):
        self.data = data
        return self

    def plot(self):
        ret = self.data[self.seq[0].l.value, self.seq[1].l.value, self.seq[2].l.value]
        #self.lastplots = ret
        #oglad.init_curves(len(ret))
        #oglad.graph_update(ret)
        return ret




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
        self.l0 = QtGui.QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        #self.image = pg.PlotDataItem()
        #self.image = pg.ImageItem()
        self.plot_viewbox = self.plot_widget.plotItem.getViewBox()
        #self.plot_viewbox.addItem(self.image)
        self.l0.addWidget(self.plot_widget)
        self.cw.setLayout(self.l0)
        self.curves = []
        self.sliders: List[Slider] = []
        self.setWindowTitle('SliceDisplay')
        self.show()
        self.activateWindow()
        self.children = []
        self.parents = []
        self.lines = {}
        self.dim = 3
        self.histogram_lut = pg.HistogramLUTWidget()
        self.histogram_lut.item.gradient.loadPreset('flame')

    def insert(self, ps:SliceSequence):
        for s in self.sliders:
            self.l0.removeWidget(s.widget)
        self.ps = ps
        self.sliders = ps.sliders
        self.dim = 3-len(self.sliders)
        psliders = []
        for p in self.parents:
            psliders.extend(p.sliders)
        for s in self.sliders:
            if s not in psliders:
                self.l0.addWidget(s.widget)
                s.onUpdate = lambda : self.plot()
        if self.dim == 1:
            self.image = pg.PlotDataItem()
            self.plot_viewbox.addItem(self.image)
            self.graph_update = lambda data : self.image.setData(data)
        elif self.dim == 2:
            self.image = pg.ImageItem()
            self.plot_viewbox.addItem(self.image)
            self.histogram_lut.setImageItem(self.image)
            self.graph_update = lambda data : self.image.setImage(data)
        self.plot()
        return self

    def add_slider(self, dim):
        s = self.sliders[dim]
        self.l0.addWidget(s.widget)
        s.onUpdate = lambda : self.plot()       
        self.plot()

    def plot(self):
        try:
            data = self.ps.plot()
            self.graph_update(data)
        except AttributeError:
            print ('No slice sequence')
        for child in self.children:
            child.plot_once()
        for parent in self.parents:
            for l in parent.lines:
                for s in self.sliders:
                    if s not in parent.sliders and l == self:
                        parent.lines[l].setValue(s.value)
        return self

    def enable_line(self, psd2:"SliceDisplay"):
        for i,s in enumerate(psd2.sliders):
            if s not in self.sliders:
                if self.dim == 1:
                    angle = 90
                if self.dim == 2:
                    if psd2.ps.seq[0].sliders == [] or s in psd2.ps.seq[2].sliders:
                        angle = 0
                    else:
                        angle = 90
                self.lines.setdefault(psd2, pg.InfiniteLine(pos=s.value, angle=angle, pen='r', movable=True, bounds =[s.min, s.max]))
                self.plot_viewbox.addItem(self.lines[psd2])
                a = i #inaczej linia przesuwa ostatni suwak na liście, bo sigDragged bierze ostatnią wartość i
                self.lines[psd2].sigDragged.connect(lambda : psd2.sliders[a].upd(self.lines[psd2].value()))

    def plot_once(self):
        try:
            data = self.ps.plot()
            self.graph_update(data)
        except AttributeError:
            print ('No slice sequence')

           

    def connect(self, psd2:"SliceDisplay"):
        self.children.append(psd2)
        psd2.parents.append(self)



class SliceWindow(QtGui.QMainWindow):
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
        
        self.psd = SliceDisplay()
        self.psd2 = SliceDisplay()
        self.psd3 = SliceDisplay()

        self.l0.addWidget(self.psd, 0, 0, 2, 2)
        self.l0.addWidget(self.psd2, 0, 2, 1, 1)
        self.l0.addWidget(self.psd3, 1, 2, 1, 1)
        self.cw.setLayout(self.l0)
        self.setWindowTitle('SliceWindow')
        self.show()
        self.activateWindow()


    def setdata(self, V):
        self.sliderx = Slice(Slider(0,0,V.shape[0]))
        self.slidery = Slice(Slider(V.shape[1]//2,0,V.shape[1]))
        self.sliderz = Slice(Slider(V.shape[2]//2,0,V.shape[2]))

        ps = SliceSequence(self.sliderx,Slice(0),Slice(0))
        ps2 = SliceSequence(self.sliderx,self.slidery,Slice(0))
        ps3 = SliceSequence(self.sliderx,Slice(0),self.sliderz)

        self.psd.connect(self.psd2)
        self.psd.connect(self.psd3)


        ps = ps.setdata(V)
        ps2 = ps2.setdata(V)
        ps3 = ps3.setdata(V)

        self.psd.insert(ps)
        self.psd2.insert(ps2)
        self.psd3.insert(ps3)

        self.psd.enable_line(self.psd2)
        self.psd.enable_line(self.psd3)
