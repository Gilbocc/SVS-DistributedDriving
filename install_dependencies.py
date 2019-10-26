import os

#conda dependencies
os.system('conda install -c conda-forge tensorflow=1.4')
os.system('conda install h5py')
os.system('conda install keras')
os.system('conda install jupyter')
os.system('conda install pandas')
os.system('conda install numpy')
os.system('conda install scipy')
os.system('conda install -c conda-forge matplotlib=2.1.2')
os.system('conda install -c conda-forge opencv')
#pip dependencies
os.system('python -m pip install --upgrade pip')
os.system('pip install image')
os.system('pip install keras_tqdm')
os.system('pip install msgpack-rpc-python')
os.system('pip install django-ipware')
os.system('pip install requests')