'''
moduł obsługi symulacji xmds
'''
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import os
import time
import dill

def tree_iter(e,depth=()):
    'tree walk as iterator, returns (path, leaf)'
    for ce in e:
        subdepth = depth + (ce.tag,)
        if ce.text:
            yield subdepth,ce.text
        yield from tree_iter(ce,subdepth)
        if ce.tail:
            yield subdepth,ce.tail

def find_mg_files(xsilfilepath):
    'znajdź na ślepo pliki _mgN.dat'
    fnames=[re.sub(r'.xsil$', f'_mg{i}.dat',str(xsilfilepath)) for i in range(3)]
    assert all(Path(f).exists() for f in fnames)
    return fnames

def xtract_params_from_filename(xsilfilepath):
    'odcyfruj parametry z nazwy pliku'
    # podziel kropkami po których nie ma cyfry
    spars=re.split(r'\.(?=\D)',str(xsilfilepath))
    # dopasuj nazwa_parametru_10.0
    matchs=[re.fullmatch(r'(\w+)_(\d+(\.\d*)?)', par) for par in spars]
    return {m[1]:float(m[2]) for m in matchs if m}

def xtract_timestamp_from_filename(xsilfilepath):
    'odcyfruj parametry z nazwy pliku'
    # podziel kropkami po których nie ma cyfry
    spars=re.split(r'\.(?=\D)',str(xsilfilepath))
    return spars[1]

class Result:
    '''
    wyniki jednej symulacji (jeden zestaw parametrów, potencjalnie wiele plików mgN.dat)

        xml_root - drzewo xml
        info - wartości parametrów run-time (long string)
        tranverse_dimensions - np. {'z': {'domain': (-1, 1), 'lattice': 256}}
        output_basis - np. {'z': 256, 't': 250}
        momentgroups - List[XSilMomentGroup] (klasa nizej - opisy plików binarnych)
        assigments - long string wszystkie przypisania (C++) z .xmds
        spacevars - lista kierunków przestzeni np.  ['t', 'z']
        cmplxfields - lista pól zespolonych - wyników symulacji

    uwaga:xmds zapisuje błędy symulacji dla każdego pola, pomijamy (vide skiperror)
    '''
    def __init__(self, xsilfilepath):
        #self.findfiles(xsilfilepath)
        self.xsilfilepath=xsilfilepath
        self.params = xtract_params_from_filename(xsilfilepath)
        self.loadxsil(xsilfilepath)

    def loadxsil(self, xsilfilepath):
        '''załaduj .xsil z opisem wyników i wyciągnij różne informacje'''
        tree = ET.parse(xsilfilepath)
        root = tree.getroot() # drzewo xml
        self.info=next(root.iter('info')).text
        prop_dim=root.find('./geometry/propagation_dimension').text
        steps=int(root.find('./sequence/integrate').attrib['steps'])
        self.tranverse_dimensions={}
        for ad in [e.attrib for e in
                   root.findall('./geometry/transverse_dimensions/dimension')]:
            kn=ad.pop('name')
            rdct = {}
            locals().update(self.params)
            for k, v in ad.items():
                if k in ('transform',):
                    rdct[k] = v
                    continue
                try:
                    rdct[k] = eval(v)
                except Exception as e:
                    print(f'k={k}, v={v} eval failed')
                    rdct[k] = v
            self.tranverse_dimensions[kn]=rdct
        output_basis=root.find('./output/group/sampling').attrib['basis']

        matchs = [re.fullmatch(r'(\w+)(\((\d+)\))?', par) for par in output_basis.split(' ')]
        self.output_basis = {m[1]: int(m[2]) if m[2] else self.tranverse_dimensions[m[1]]['lattice']
                             for m in matchs if m}
        self.output_basis[prop_dim]=steps
        self.xml_root = root
        self.momentgroups = [XSilMomentGroup(e, xsilfilepath) for e in root.iter('XSIL')]
        # self.globals=root.find('./features/globals').text
        self.assigments = [(pth,txt) for pth, txt in tree_iter(root) if '=' in txt]
        self.spacevars = self.momentgroups[0].spacevars
        self.cmplxfields = [f for m in self.momentgroups for f in m.get_cmplxflds()]

    def shortdescr(self):
        'string krotko opisujacy rezultaty'
        sdct=' '.join(f'{k}({v})' for k,v in self.output_basis.items())
        scmplxfields=' '.join(self.cmplxfields)
        msizeMB = np.prod(list(self.output_basis.values())) * 16 / 1024 / 1024
        return f'{sdct}: {scmplxfields} ({msizeMB:5.1f}MB/cmplx3dmatrix)'

    def load(self):
        'ładuj pliki binarne z wynikami'
        results={}
        for mg in self.momentgroups:
            print('loading', mg.shortdescr())
            mg.load(results)
        self.results=results
        self.__dict__.update(results)
        return results
    def print_assigments(self):
        'wypisz wszystkie przyrównania z xml-a'
        for pth, txt in self.assigments:
            print('-'*10+' '+'/'.join(pth))
            txt2='\n'.join(ln.strip() for ln in txt.strip().split('\n') if len(ln.strip())>1)
            print(txt2)


class XSilMomentGroup:
    '''
    opis jedengo pliku binarnego _mgN.dat odczytany z .xsil
        - name np. 'moment_group_1'
        - path - plik binarny
        - params np. {'n_independent': '2'}
        - vars - lista kierunki przestzreni i pola (składowe R/I)
        - cmplxcompose - lista trójek do składania pól zespolonych np. [('ER', 'EI', 'E'),...]
        - dims np. [250, 256] - shape wyników
        - spacevars np. ['t', 'z']
        - varsxml, data - poddrzewa xml
    '''
    def __init__(self, xe: ET.Element, xsilfilepath: Path):
        self.name=xe.attrib['Name']
        self.params={e.attrib['Name']:e.text for e in xe.iter('Param')}
        arrs={a.name:a for a in [XSilMGArray(a) for a in xe.iter('Array')]}
        self.varsxml = arrs['variables']
        self.vars = self.varsxml.text.split(' ')
        suspects = [(n,n.replace('real','imag'),n.replace('real','')) for n in self.vars if 'real' in n]
        suspectsRI = [(n, n[:-1]+'I', n[:-1]) for n in self.vars if n[-1]=='R']
        self.cmplxcompose = [nnn for nnn in suspects+suspectsRI if nnn[1] in self.vars]
        self.data = arrs['data']
        self.dims = self.data.dims[:-1]
        #self.path = xsilfilepath.parent/self.data.text
        self.path = xsilfilepath.parent/Path(self.data.text).name
        self.spacevars = self.vars[:len(self.dims)]
    def get_cmplxflds(self, skiperror=True):
        'nazwy pól zespolonych'
        return [n for _,_,n in self.cmplxcompose if not skiperror or not n.startswith('error_')]
    def shortdescr(self):
        'wygeneruj krótki opis'
        sdims='*'.join(f'{d}' for d in self.dims)
        svars=', '.join(self.vars)
        return f'{self.name}: file_exists={self.path.exists()} {sdims} {svars}'
    def load(self, results: dict, skiperror=True):
        'załaduj dane z pliku'
        data=XilLoader(self.path).load(len(self.vars))
        xdims=self.dims+[np.prod(self.dims)]*(len(self.vars)-len(self.dims))
        for i, (n, l, d) in enumerate(zip(self.vars, xdims, data)):
            if n.startswith('error_') and skiperror:
                continue
            assert len(d)==l, f'data size mismatch {l}!={len(d)}'
            if i<len(self.dims):
                if n in results:
                    pass # TODO: check
                else:
                    results[n]=d
            else:
                results[n]=d.reshape(self.dims)
        for r, i, c in self.cmplxcompose:
            if r in results and i in results:
                results[c]=results[r]+1j*results[i]


class XSilMGArray:
    'klasa pomocnicza: rozkodowanie elementu XSIL/Array'
    def __init__(self, a: ET.Element):
        self.name = a.attrib['Name']
        self.type = a.attrib['Type']
        self.dims = [int(e.text.strip()) for e in a.iter('Dim')]
        stream = next(a.iter('Stream'))
        metalink=next(stream.iter('Metalink'))
        self.__dict__.update(metalink.attrib)
        self.text=metalink.tail.strip()


class XilLoader:
    'pomocnicze ładowanie pliku binarnego z XMDS'
    def __init__(self,filepath):
        self.filepath=filepath
    def load(self, max_arrays=50):
        with open(self.filepath,'rb') as infile:
            res=[]
            while infile and len(res)<max_arrays:
                datalen=np.fromfile(infile,dtype='<u8',count=1)[0]
                data=np.fromfile(infile,dtype='<f8',count=datalen)
                res.append(data)
            return res


class ParamsClass:
    'prototyp zbioru parametrów'

    # class cproperty(property): pass
    def __init__(self):
        self.Atimestamp = f'{int(time.time() % 1e8):X}'

    def toDict(self, bprint=False):
        dl = list(self.__dict__.items()) + list(self.__class__.__dict__.items())
        # d={k: v for k, v in dl if k[:2] != '__' and isinstance(v, (int, complex, float))}
        d = {};
        osl = []
        for k, v in dl:
            if not k.startswith('_'):
                osl.append(f'   {k}:{v}')
                if isinstance(v, (int, float, str)):
                    d[k] = v
                elif isinstance(v, complex):
                    d.update(self.fromProperty(k))
                elif isinstance(v, self.cproperty):
                    d.update(self.fromProperty(k))
        if bprint and os:
            print('\n'.join(osl))
        return d

    def grid_zt(g):
        z = np.linspace(-g.zradious, g.zradious, num=g.zsteps)
        t = np.linspace(0, g.tmax * (1 + 2 / g.tsteps), num=g.tsteps + 2)
        return z, t

    def fromProperty(self, prop):
        'generate Im and Re params from property'
        v = getattr(self, prop)
        return {prop + '_re': np.round(np.real(v), 5), prop + '_im': np.round(np.imag(v), 5)}

root_path = Path(__file__).parent.relative_to(Path().resolve())

class Simulation:
    'kompilacja i uruchamianie xmds w ramach windows subsystem linux'
    xmds_file_out = root_path/'xmds_out/mod.xmds'
    rtparams={}

    def write_xmds(self, template_file, params: dict):
        'podstw wartosci do templat-u xmds, zapisz xmds_file_out'
        self.params=params
        with open(template_file, 'r') as file:
            template = file.read()
        print(params)
        with open(self.xmds_file_out, 'w') as ofile:
            ofile.write(template.format(**self.params))
        return self
    def compile(self, outfile=None):
        'uruchom kompilację'
        print('compiling...')
        fpath =self.xmds_file_out.name
        cmd = f'cd {self.xmds_file_out.parent} && wsl xmds2 {fpath}'
        if outfile:
            cmd += f' -o ../{outfile}'
        print(cmd)
        return os.system(cmd)
    def run(self, rtparams, simname, wait=False):
        'uruchom symulację'
        self.rtparams=rtparams
        self.simname=simname
        parstr=' '.join(f'--{k}={v}' for k,v in self.rtparams.items())
        with open(self.xmds_file_out.parent/'run.sh','w') as runfile:
            oredir = '>' if wait else '2>&1 | tee'
            cmd=f'mpirun -np 2 {self.simname} {parstr} {oredir} sim.log'
            runfile.write(cmd)
        if wait:
            ret =  os.system(f'cd {self.xmds_file_out.parent} && wsl ./run.sh')
            if ret:
                with open(self.xmds_file_out.parent/'sim.log','r') as f:
                    print(f.read())
            return ret
        else:
            return os.system(f'cd {self.xmds_file_out.parent} && start wsl ./run.sh')
    def run_load(self, rpars:ParamsClass, simname):
        ret = self.run(rpars.toDict(True), simname, wait=True)
        return self.load(rpars)

    def load(self, rpars:ParamsClass):
        path=self.xmds_file_out.parent
        patt = f'*timestamp_{rpars.Atimestamp}.xsil'
        patt2 = f'*timestamp_{rpars.Atimestamp}*.xsil'
        flist = list(path.glob(patt)) + list(path.glob(patt2))
        assert len(flist) == 1, f'brak lub za dużo wyników - pliki:{flist}\n' \
                f'run code = {ret}, pattern = {path.absolute()}/{patt}'
        r = Result(flist[0])
        #print(r.shortdescr())
        r.load()
        return r



class SimInputData:
    """
    pola podowane do symulacji
    vide load_from_binary_file w .xmds
    forma klasy dla uporządkowania

    UWAGA: pola muszą być typu complex
    """
def init_path():
    path = root_path/'init_data'
    path.mkdir(parents=True, exist_ok=True)
    return path

def clear_init():
    path = root_path/'init_data/'
    flist = list(path.glob('*.bin'))
    for f in flist:
        os.remove(f)

def load_sim(n):
    path = root_path/'xmds_out/'
    flist = list(path.glob('2eit1t.Atimestamp*.xsil'))
    print (flist[n])
    r = Result(flist[n])
    r.load()
    clear_init()
    return r

def load_output(n, name):
    '''
    wczytywanie wyników symulacji
    '''
    path = root_path/'xmds_out/'
    flist = list(path.glob(name+'.Atimestamp*.xsil'))
    flist2 = list(path.glob(name+'.Atimestamp*.pkl'))
    print (flist[n])
    print (flist2[n])
    r = Result(flist[n])
    r.load()
    timestamp = xtract_timestamp_from_filename(r.xsilfilepath)
    with open(flist2[n], 'rb') as file:
        sid = dill.load(file, ignore = True)
    return sid, r