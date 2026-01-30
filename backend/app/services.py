from sqlalchemy.orm import Session
from domain import schemas
from infra.db import models

class ItemService:
    def __init__(self, db: Session):
        self.db = db

    def get_items(self, skip: int = 0, limit: int = 100):
        return self.db.query(models.ItemModel).offset(skip).limit(limit).all()

    def create_user_item(self, item: schemas.ItemCreate, user_id: int):
        db_item = models.ItemModel(**item.model_dump(), owner_id=user_id)
        self.db.add(db_item)
        self.db.commit()
        self.db.refresh(db_item)
        return db_item
