'''
moduł do wyświetlania przekrojów przez dane 3d za pomocą pyqtgraph
'''
import time
from typing import List
from PyQt5 import QtCore
import pyqtgraph as pg
#from pyqtgraph.Qt import QtGui
import PyQt5.QtWidgets as QtGui
from PyQt5.QtGui import QTransform
# from pyqtgraph.Qt.QtGui import QMainWindow
import numpy as np
import xmds
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

    def setMaximum(self, v):
        self.max = v
        self.slider.setRange(self.min, self.max)
        self.num.setRange(self.min, self.max)

    def setMinimum(self, v):
        self.min = v
        self.slider.setRange(self.min, self.max)
        self.num.setRange(self.min, self.max)

    def setValue(self, v):
        self.slider.setValue(v)
        self.num.setValue(v)

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



class subobj(object):
    pass
class Oglad(pg.QtCore.QThread):
    'wyswietla przekroje danych z xmds'
    status = pg.QtCore.Signal(object)
    #newData = pg.QtCore.Signal(object)
    def __init__(self):
        super(self.__class__, self).__init__()
        #self.results=results
        #self.ui = subobj()
        self.autoscale = True
        self.lastidx=[0]*3
        #self.init_win()
    def init_win(self):
        "otwórz okienko + regulacje"
        self.win = QtGui.QMainWindow()
        self.win.resize(800, 600)
        self.win.show()
        self.win.setWindowTitle('xmds - ogląd')
        self.cw = QtGui.QWidget()
        self.win.setCentralWidget(self.cw)
        self.status.connect(lambda s: self.win.statusBar().showMessage(s))
        self.l = QtGui.QGridLayout()
        self.cw.setLayout(self.l)
        self.l.setSpacing(0)

        self.plot_widget = pg.PlotWidget()
        self.plot_viewbox = self.plot_widget.plotItem.getViewBox()
        self.l.addWidget(self.plot_widget, 0, 0, 4, 1)
        self.pimg = pg.ImageItem()
        self.plot_viewbox.addItem(self.pimg)
        
        self.histogram_lut = pg.HistogramLUTWidget()
        self.histogram_lut.item.gradient.loadPreset('flame')
        self.l.addWidget(self.histogram_lut, 0, 1, 1, 2)

        self.histogram_lut.setImageItem(self.pimg)

        
        # self.xctrl = QtGui.QSpinBox(self.cw)
        # self.l.addWidget(self.xctrl, 1, 1)
        # self.xctrl.valueChanged.connect(self.graph_update)
        self.xctrl = Slider(0,0,1)
        self.l.addWidget(self.xctrl.widget, 1, 1)
        self.xctrl.slider.valueChanged.connect(self.graph_update)

        self.sel_direction = QtGui.QComboBox(self.cw)
        self.l.addWidget(self.sel_direction, 1, 2)
        self.sel_direction.currentIndexChanged.connect(self.change_direction)
        

        self.sel_field = QtGui.QComboBox(self.cw)
        self.l.addWidget(self.sel_field, 2, 1)
        self.sel_field.currentIndexChanged.connect(self.graph_update)

        self.sel_func = QtGui.QComboBox(self.cw)
        self.l.addWidget(self.sel_func, 2, 2)
        self.sel_func.addItems(['abs','angle','real','imag'])
        self.sel_func.currentIndexChanged.connect(self.graph_update)

        self.status.emit('init_win done')
        return self

    def set_result(self, results: xmds.Result):
        'ustaw dane do wyswietlania'
        self.results = results
        self.sel_direction.addItems(self.results.spacevars)
        self.sel_field.addItems(self.results.cmplxfields)
        # TODO: rozsadniej ustawiaj przekroj poczatkowy
        data_name = self.sel_field.currentText()
        data = getattr(self.results, data_name).take(-1, axis=0)
        self.status.emit('set_result done')
        self.change_direction()
        self.plot_viewbox.autoRange()
        return self

    def change_direction(self):
        'zdrzenie na zmiane kirunku przekroju'
        try:
            axis = self.sel_direction.currentIndex()
            var = self.results.spacevars[axis]
            dshape = getattr(self.results, var).shape
            lbls=self.results.spacevars[:]
            lbls.pop(axis)
            print(f'change_direction axis={axis} var={var} dshape={dshape} lbls={lbls}')
            self.plot_widget.plotItem.setLabel('left', lbls[0])
            self.plot_widget.plotItem.setLabel('bottom', lbls[1])
            idxmax=dshape[0] - 1
            self.xctrl.setMaximum(idxmax)
            self.xctrl.setValue(min(idxmax,self.lastidx[axis]))
            self.graph_update()
            self.plot_viewbox.autoRange()
        except Exception as e:
            print('change_direction exception ',e)



    def graph_update(self):
        "zmiana ktoregokolwiek parametru przekroju"
        try:
            data_name = self.sel_field.currentText()
            '''
            if self.sel_field.currentText() == 'rho12kz':
                self.results.spacevars[1] = 'kz'
                self.sel_direction.removeItem(1)
                self.sel_direction.insertItem(1, 'kz')
            else:
                self.results.spacevars[1] = 'z'
                self.sel_direction.removeItem(1)
                self.sel_direction.insertItem(1, 'z')
            '''
            idx = self.xctrl.value
            axis = self.sel_direction.currentIndex()
            data = getattr(self.results, data_name).take(idx,axis=axis)
            func = getattr(np, self.sel_func.currentText())
            self.status.emit(f'{func} {data_name} axis={axis} idx={idx}')
            data=func(data).transpose()
            #self.newData.emit(m1c.transpose() if self.transpose else m1c)
            if not self.autoscale:
                lvls = self.pimg.getLevels()
                self.pimg.setImage(data, autoLevels=False, levels=lvls)
            else:
                self.pimg.setImage(data)#, cmap=cm.coolwarm)
            self.lastidx[axis]=idx
            x,y = self.find_axes()
            self.scale_axes(x,y)
        except Exception as e:
            print('change_direction exception ',e)


    def find_axes(self):
        'znajdowanie które wymiary są na osiach wykresu'
        axis = self.sel_direction.currentIndex()
        var = self.results.spacevars[axis]
        lbls=self.results.spacevars[:]
        lbls.pop(axis)
        y = getattr(self.results, lbls[0])
        x = getattr(self.results, lbls[1])
        return x, y
            

    def scale_axes(self, x, y):
        'skalowanie osi wykresu przy użyciu QTransform, x, y - tablice będące nowymi osiami'
        transform = QTransform()
        h = self.pimg.height()
        w = self.pimg.width()
        ax_x = self.plot_widget.plotItem.getAxis('bottom')
        transform.scale((x[-1]-x[0])/w, (y[-1]-y[0])/h)
        transform.translate(w*x[0]/(x[-1]-x[0]), h*y[0]/(y[-1]-y[0]))
        self.pimg.setTransform(transform)


# %%
