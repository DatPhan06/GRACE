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
    avg_recall_retrieval = Column(Float, nullable=True)
    avg_recall_semantic = Column(Float, nullable=True)
    avg_recall_content = Column(Float, nullable=True)
    avg_recall_collab = Column(Float, nullable=True)
    status = Column(String, default="pending") # pending, completed, error
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship
    # Relationship
    results = relationship("EvaluationResultModel", back_populates="run", cascade="all, delete-orphan")
    conversations = relationship("ConversationLogModel", back_populates="run", cascade="all, delete-orphan")
    batches = relationship("BatchStepExecutionModel", back_populates="run", cascade="all, delete-orphan")

class ConversationLogModel(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    conv_id = Column(String)
    index = Column(Integer)
    status = Column(String, default="pending") # pending, summarized, retrieved, reranked, completed, error
    
    # Input Data
    dialog = Column(JSON) # The raw dialog
    target = Column(JSON) # Ground truth
    liked_movies = Column(JSON) # Extracted liked movies

    # Relationships
    run = relationship("EvaluationRunModel", back_populates="conversations")
    summarization = relationship("StepSummarizationModel", back_populates="conversation", cascade="all, delete-orphan")
    retrieval = relationship("StepRetrievalModel", back_populates="conversation", cascade="all, delete-orphan")
    reranking = relationship("StepRerankingModel", back_populates="conversation", cascade="all, delete-orphan")
    result = relationship("EvaluationResultModel", back_populates="conversation", uselist=False, cascade="all, delete-orphan")


class BatchStepExecutionModel(Base):
    __tablename__ = "batch_step_executions"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    step_type = Column(String) # summarization, retrieval, reranking
    version = Column(Integer)
    config = Column(JSON)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    run = relationship("EvaluationRunModel", back_populates="batches")


class StepSummarizationModel(Base):
    __tablename__ = "step_summarization"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversation_logs.id"))
    batch_run_id = Column(Integer, ForeignKey("batch_step_executions.id"))
    user_preferences = Column(String)
    
    conversation = relationship("ConversationLogModel", back_populates="summarization")
    batch = relationship("BatchStepExecutionModel")


class StepRetrievalModel(Base):
    __tablename__ = "step_retrieval"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversation_logs.id"))
    batch_run_id = Column(Integer, ForeignKey("batch_step_executions.id"))
    
    # Storing full candidate lists might be heavy, but useful for debugging. 
    # For now, let's store simplified versions or just IDs if mapping is available, 
    # but storing full JSON of candidates (title, year, plot) is safer for reproducibility.
    candidates = Column(JSON) 
    
    # Metadata
    semantic_count = Column(Integer)
    content_count = Column(Integer)
    collab_count = Column(Integer)

    conversation = relationship("ConversationLogModel", back_populates="retrieval")
    batch = relationship("BatchStepExecutionModel")


class StepRerankingModel(Base):
    __tablename__ = "step_reranking"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversation_logs.id"))
    batch_run_id = Column(Integer, ForeignKey("batch_step_executions.id"))
    
    model_used = Column(String)
    reranked_candidates = Column(JSON)
    recall = Column(Float, nullable=True)
    
    conversation = relationship("ConversationLogModel", back_populates="reranking")
    batch = relationship("BatchStepExecutionModel")


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    conversation_id = Column(Integer, ForeignKey("conversation_logs.id"), nullable=True) # Link to log
    
    conv_id = Column(String)
    recall = Column(Float)
    ground_truth = Column(JSON)  # Store list of strings
    recommendations = Column(JSON)  # Store list of strings
    candidate_count = Column(Integer)
    semantic_count = Column(Integer, nullable=True)
    content_count = Column(Integer, nullable=True)
    collab_count = Column(Integer, nullable=True)

    recall_retrieval = Column(Float, nullable=True)
    recall_semantic = Column(Float, nullable=True)
    recall_content = Column(Float, nullable=True)
    recall_collab = Column(Float, nullable=True)
    error = Column(String, nullable=True)

    # Relationship
    run = relationship("EvaluationRunModel", back_populates="results")
    conversation = relationship("ConversationLogModel", back_populates="result")
