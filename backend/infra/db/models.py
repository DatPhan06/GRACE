from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
import datetime

from infra.db.database import Base

class ItemModel(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    owner_id = Column(Integer, index=True)


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset = Column(String)
    sample_size = Column(Integer)
    start_index = Column(Integer)
    n_sample = Column(Integer)
    top_k = Column(Integer)
    model = Column(String)
    avg_recall = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    results = relationship("EvaluationResultModel", back_populates="run", cascade="all, delete-orphan")


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    conv_id = Column(String)
    recall = Column(Float)
    ground_truth = Column(JSON)  # Store list of strings
    recommendations = Column(JSON)  # Store list of strings
    candidate_count = Column(Integer)
    error = Column(String, nullable=True)

    # Relationship
    run = relationship("EvaluationRunModel", back_populates="results")
