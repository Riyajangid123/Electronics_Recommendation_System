from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class Product(BaseModel):

    name: str

    image: Optional[HttpUrl] = None

    price: Optional[float] = None

    ratings: Optional[float] = None

    link: Optional[HttpUrl] = None


class Recommendation(BaseModel):

    query: str

    recommendations: List[Product]