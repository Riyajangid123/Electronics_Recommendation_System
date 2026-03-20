from pydantic import BaseModel,HttpUrl
from typing import Optional

class Product(BaseModel):
    name: str
    sub_category: Optional[str] = None
    main_category: Optional[str] = None
    image: Optional[HttpUrl] = None
    link: Optional[HttpUrl] = None
    ratings: Optional[float] = None
    no_of_ratings: Optional[int] = None
    discount_price: Optional[float] = None
    actual_price: Optional[float] = None

class Recommendation(BaseModel):
    query: str  
    recommendations: list[Product]  
