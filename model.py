from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()


class Attendance(Base):
    __tablename__ = 'Attendance'
    id = Column(Integer, primary_key=True)
    employeeID = Column(Integer)
    name = Column(String(250), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String(50), nullable=False)
    status = Column(String(10), nullable=False)


class Timekeeping(Base):
    __tablename__ = 'Timekeeping'
    id = Column(Integer, primary_key=True)
    employeeID = Column(Integer)
    name = Column(String(250), nullable=False)
    date = Column(Date, nullable=False)
    checkIn = Column(String(50))
    checkOut = Column(String(50))
    last = Column(String(50))
    work = Column(Integer)
    rest = Column(Integer)
    OT = Column(Integer)


class Employee(Base):
    __tablename__ = 'Employee'
    id = Column(String(10), primary_key=True)
    name = Column(String(250), nullable=False)


# Link to database
engine = create_engine('sqlite:///Attendance.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
