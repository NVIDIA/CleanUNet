FROM nvcr.io/nvidia/pytorch:20.12-py3
RUN apt-get update --fix-missing

RUN pip install pillow==6.2.0
RUN pip install torchaudio==0.8.0
RUN pip install inflect==4.1.0
RUN pip install scipy==1.5.0
RUN pip install tqdm
RUN pip install pesq
RUN pip install pystoi