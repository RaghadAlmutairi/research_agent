from sqlalchemy import String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from uuid import uuid4
from datetime import datetime


class Base(DeclarativeBase):
    pass


class CrawlJob(Base):

    __tablename__ = "crawl_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    url: Mapped[str] = mapped_column(String(2000))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    source_type: Mapped[str] = mapped_column(String(50), default="firecrawl")
    page_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pages: Mapped[list["ScrapedPage"]] = relationship(
        "ScrapedPage", back_populates="crawl_job"
    )


class ScrapedPage(Base):

    __tablename__ = "scraped_pages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    url: Mapped[str] = mapped_column(String(2000))
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    crawl_job_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("crawl_jobs.id"),
        nullable=True,
    )

    crawl_job: Mapped["CrawlJob | None"] = relationship(
        "CrawlJob", back_populates="pages"
    )


class Competitor(Base):

    __tablename__ = "competitors"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    website: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(255), nullable=True)


class CustomerReview(Base):

    __tablename__ = "customer_reviews"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    platform: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
