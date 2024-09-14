FROM python:3.9

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /opt/backend/phantom/app

WORKDIR /opt/backend/phantom/app

ADD requirements.txt .

RUN pip3 install -r requirements.txt

CMD python3 start.py