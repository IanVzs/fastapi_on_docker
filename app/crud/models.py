from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from .database import Base

Base = declarative_base()

class CRUD(Base):
    __tablename__ = 'crud_current'
    __table_args__ = (
        UniqueConstraint('gaid'),
        Index('country_idx', 'country'),
        Index('created_idx', 'created')
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    country = Column(String(3), nullable=False)
    data_source = Column(String(20), nullable=False)
    gaid = Column(String(36), nullable=False)
    status = Column(String(10), nullable=False)
    lock_status = Column(String(20), nullable=False)
    lock_start = Column(DateTime, default=None)
    device_json = Column(Text, nullable=False)
    device_json_new = Column(Text)
    created = Column(DateTime, nullable=False)
    updated = Column(DateTime, nullable=False)
    model = Column(String(54), default=None)
