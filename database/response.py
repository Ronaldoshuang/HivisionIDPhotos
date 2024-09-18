# _*_ coding: utf-8 _*_
from typing import Union

from pydantic import BaseModel


class CRUD(BaseModel):
    data: Union[str, list, dict] = None
    msg: Union[str, tuple] = None  # Exception的返回参数args是一个tuple
    status: bool = True
