-- Create Department Table
CREATE TABLE Department (
    DepartmentId INT PRIMARY KEY,
    DepartmentName NVARCHAR(255)
);

-- Create GroupDC Table
CREATE TABLE GroupDC (
    GroupDCId INT PRIMARY KEY,
    DepartmentId INT,
    GroupDCName NVARCHAR(255),
    FOREIGN KEY (DepartmentId) REFERENCES Department(DepartmentId)
);

-- Create Author Table
CREATE TABLE Author (
    AuthorId INT PRIMARY KEY,
    AuthorName NVARCHAR(255),
    DepartmentId INT,
    GroupDCId INT,
    FOREIGN KEY (DepartmentId) REFERENCES Department(DepartmentId),
    FOREIGN KEY (GroupDCId) REFERENCES GroupDC(GroupDCId)
);

-- Create Job Table
CREATE TABLE Job (
    JobId INT PRIMARY KEY,
    JobName NVARCHAR(255)
);

-- Create Tool Table
CREATE TABLE Tool (
    ToolId INT PRIMARY KEY,
    ToolName NVARCHAR(255),
    ToolDescription TEXT
);

-- Create Jidouka Table
CREATE TABLE Jidouka (
    JidoukaId BIGINT PRIMARY KEY,
    ProductApply NVARCHAR(255),
    ImprovementName NVARCHAR(255),
    SoftwareUsing NVARCHAR(255),
    Description NVARCHAR(255),
    Video TEXT,
    DetailDocument TEXT,
    TotalJobApplied INT,
    TotalTimeSaved INT,
    DateCreate DATETIME,
    JobId INT,
    AuthorId INT,
    DepartmentId INT,
    GroupDCId INT,
    FOREIGN KEY (JobId) REFERENCES Job(JobId),
    FOREIGN KEY (AuthorId) REFERENCES Author(AuthorId),
    FOREIGN KEY (DepartmentId) REFERENCES Department(DepartmentId),
    FOREIGN KEY (GroupDCId) REFERENCES GroupDC(GroupDCId)
);

-- Create JidoukaTool Table
CREATE TABLE JidoukaTool (
    JidoukaId BIGINT,
    ToolId INT,
    PRIMARY KEY (JidoukaId, ToolId),
    FOREIGN KEY (JidoukaId) REFERENCES Jidouka(JidoukaId),
    FOREIGN KEY (ToolId) REFERENCES Tool(ToolId)
);
