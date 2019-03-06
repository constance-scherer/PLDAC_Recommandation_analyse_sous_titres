import socks
import socket
import time
import stem
import stem.process
from stem.control import Controller
from stem import Signal
import http.client
import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import logging
import threading
from threading import Thread
from queue import Queue, PriorityQueue
import time
import sys
import traceback

LOWESTPRIORITY=100
MAXIMALPRIORITY=100

class SocksiPyConnection(http.client.HTTPConnection):

    def __init__(self, proxytype, proxyaddr, proxyport = None, rdns = True, username = None, password = None, *args, **kwargs):
        self.proxyargs = (proxytype, proxyaddr, proxyport, rdns, username, password)
        http.client.HTTPConnection.__init__(self, *args, **kwargs)

    def connect(self):
        self.sock = socks.socksocket()
        self.sock.setproxy(*self.proxyargs)
        if isinstance(self.timeout, float):
            self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))

class SocksiPyHandler(urllib.request.HTTPHandler):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kw = kwargs
        urllib.request.HTTPHandler.__init__(self)

    def http_open(self, req):
        def build(host, port=None, strict=None, timeout=0):
            conn = SocksiPyConnection(*self.args, host=host, port=port, strict=strict, timeout=timeout, **self.kw)
            return conn
        return self.do_open(build, req)

class TorConnection(object):

    CUR_PORT=9220
    NBTRIALS=10

    def __init__(self,timeout=30):
        self.tor_proc=None
        self.port=None
        self.timeout=timeout
        self.init_con(timeout)
        self.opener = urllib.request.build_opener(SocksiPyHandler(socks.PROXY_TYPE_SOCKS4, 'localhost', self.port))
        self.failslist=[]


    def init_con(self,timeout=30):
        cpt=0
        while True:
            try:
                logging.debug("Attempt to get new port for Tor")
                if cpt % 3 == 0: self.port=TorConnection.get_new_port()
                logging.debug("Getting new port :%d" %(self.port,))
                self.tor_proc=stem.process.launch_tor_with_config(config={'SocksPort':str(self.port),'DataDirectory':"/tmp/"+str(self.port)},timeout=timeout,init_msg_handler=self.print_b)
                logging.info("Tor connection initiated (port :%d)" %(self.port,))
                return True
            except OSError as e:
                logging.debug("TorConnection.init_con : %s "% (e,))
            cpt+=1

    def print_b(self,x):
        logging.info("%d --- %s"% (self.port,x))

    def get_request(self,req,timeout=None):
        cpt=0
        code=-1
        if not timeout:
            timeout=self.timeout
        while cpt<TorConnection.NBTRIALS:
            cpt+=1
            try:
                # Pour accéder à Metacritic sans uner erreur 403, on a besoin de cet user_agent
                user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
                # On l'ajoute aux headers 
                self.opener.addheaders = [('User-agent', user_agent)]
                h=self.opener.open(req,timeout=timeout)
                code=h.code
                page=h.read()
                return page,code
            except (urllib.error.URLError,socket.timeout) as e:
                code=e.code
                #logging.info("get_request: Error for %d times, %s" %(cpt,req.get_full_url()))
            except (http.client.IncompleteRead,socket.timeout) as e:
                logging.info("get_request: TIMED OUT for %d times, %s" %(cpt,req.get_full_url()))
            except Exception as e:
                logging.info("get_request Exception: %s"% (str(e),))

        #logging.error("get_request : number trials excepted for %s" % (req.get_full_url(),))
        self.failslist.append(req)
        return None,code

    def close(self):
        self.tor_proc.terminate()

    @staticmethod
    def get_new_port():
        TorConnection.CUR_PORT+=1
        return TorConnection.CUR_PORT

class WorkerDL(threading.Thread):

    def __init__(self,tasks,con):
        threading.Thread.__init__(self)
        self.tasks=tasks
        self.con=con
        self._stop=threading.Event()
        self.daemon=False
        self.start()

    def run(self):
        while(not self._stop.isSet()):
            prior,url,cb,kwargs = self.tasks.get()
            if not url:
                break
            try:
                timeout=None
                if 'timeout' in kwargs:
                    timeout=kwargs['timeout']
                logging.debug("requete : %s" %(url.get_full_url(),))
                page,code=self.con.get_request(url,timeout)
                
                #cb(page=page,code=code,url=url,**kwargs)

            except Exception as e :
                logging.error("Exception: Worker.run")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,limit=10, file=sys.stderr)
            finally:
                self.tasks.task_done()

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return not self._stop.isSet()

class MultiDLManager:

    def __init__(self,num_con=2,num_threads=10,maxqueue=100000,timeout=30):
        self.tasks=PriorityQueue(maxqueue)
        self.workers=[]
        self.cons=[]
        jobs=[]
        for _ in range(num_con):
            #t=threading.Thread(target=self._create_con(num_threads,timeout))
            #t.start()
            #jobs.append(t)
        #for j in jobs:
            #j.join()
            self._create_con(num_threads,timeout)

    def get_request(self,req,timeout=None):
        
        return self.cons[0].get_request(req,timeout)

    def _create_con(self,num_threads,timeout=30):
        con=TorConnection(timeout)
        self.cons.append(con)
        for _ in range(num_threads): self.workers.append(WorkerDL(self.tasks,con))

    def add_task(self,url,cb,**kwargs):
        if "priority" not in kwargs:
            self.tasks.put((LOWESTPRIORITY,url,cb,kwargs))
        else:
            prior=kwargs.pop("priority")
            self.tasks.put((prior,url,cb,kwargs))

    def wait_completion(self):
        self.tasks.join()

    @property
    def len(self):
        return self.tasks.qsize()

    def stop(self):
        for w in self.workers:
            w.stop()
            self.tasks.put((MAXIMALPRIORITY,None,None,None))

    def qsize(self):
        return self.tasks.qsize()

    def close(self):
        self.stop()
        for c in self.cons:
            c.close()

    def hasAlive(self):
        return len([w for w in self.workers if w.is_alive()])
