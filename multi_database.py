import sqlite3
import os

DATABASE_FOLDER = "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/sqlite_databases"

if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

def setup_database(db_name, table_name, create_table_sql, data):
        
    db_path = os.path.join(DATABASE_FOLDER, db_name)  
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute(create_table_sql)

    if data:
        placeholders = ",".join(["?" for _ in data[0]]) 
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        cur.executemany(query, data)

    conn.commit()
    conn.close()
    print(f"Table {table_name} setup complete in {db_path}!")

# ---------------------- PLAYERS DATABASE ----------------------
setup_database(
    "players.db",
    "PLAYERS",
    """
    CREATE TABLE IF NOT EXISTS PLAYERS(
        player_id INTEGER PRIMARY KEY,
        name VARCHAR(100),
        team_id INTEGER,
        position VARCHAR(50),
        goals INT,
        assists INT,
        matches INT,
        FOREIGN KEY (team_id) REFERENCES TEAMS(team_id)
    );
    """,
    [
    (1, 'Lionel Messi', 1, 'Forward', 21, 14, 30),
    (2, 'Cristiano Ronaldo', 2, 'Forward', 24, 12, 32),
    (3, 'Neymar Jr.', 3, 'Forward', 17, 19, 25),
    (4, 'Kevin De Bruyne', 4, 'Midfielder', 8, 22, 29),
    (5, 'Robert Lewandowski', 5, 'Forward', 26, 5, 34),
    (6, 'Kylian Mbappe', 6, 'Forward', 27, 9, 33),
    (7, 'Erling Haaland', 4, 'Forward', 30, 6, 35),
    (8, 'Bruno Fernandes', 7, 'Midfielder', 10, 15, 31),
    (9, 'Mohamed Salah', 8, 'Forward', 23, 13, 32),
    (10, 'Luka Modric', 9, 'Midfielder', 5, 18, 28),
    (11, 'Harry Kane', 10, 'Forward', 25, 8, 33),
    (12, 'Vinicius Jr.', 9, 'Forward', 15, 11, 30),
    (13, 'Jude Bellingham', 9, 'Midfielder', 9, 14, 29),
    (14, 'Antoine Griezmann', 11, 'Forward', 18, 10, 31),
    (15, 'Romelu Lukaku', 12, 'Forward', 21, 7, 32),
    (16, 'Son Heung-min', 13, 'Forward', 19, 9, 34),
    (17, 'Bernardo Silva', 4, 'Midfielder', 6, 17, 30),
    (18, 'Bukayo Saka', 14, 'Midfielder', 12, 16, 31),
    (19, 'Marcus Rashford', 7, 'Forward', 20, 5, 30)
    ]
)

# ---------------------- TEAMS DATABASE ----------------------
setup_database(
    "teams.db",
    "TEAMS",
    """
    CREATE TABLE IF NOT EXISTS TEAMS(
        team_id INTEGER PRIMARY KEY,
        team_name VARCHAR(50),
        stadium_id INTEGER,
        wins INT,
        losses INT,
        draws INT,
        points INT,
        finance_id INTEGER,
        FOREIGN KEY (stadium_id) REFERENCES STADIUMS(stadium_id),
        FOREIGN KEY (finance_id) REFERENCES FINANCIALS(finance_id)
    );
    """,
    [
    (1, 'Inter Miami', 1, 15, 5, 3, 48, 1),
    (2, 'Al-Nassr', 2, 17, 6, 1, 52, 2),
    (3, 'Al-Hilal', 3, 14, 8, 2, 44, 3),
    (4, 'Manchester City', 4, 28, 3, 7, 91, 4),
    (5, 'FC Barcelona', 5, 27, 5, 6, 87, 5),
    (6, 'Paris Saint-Germain', 6, 31, 1, 6, 99, 6),
    (7, 'Manchester United', 7, 22, 9, 7, 73, 7),
    (8, 'Liverpool', 8, 26, 4, 8, 86, 8),
    (9, 'Real Madrid', 9, 30, 2, 6, 96, 9),
    (10, 'Bayern Munich', 10, 29, 1, 8, 95, 10),
    (11, 'Atletico Madrid', 11, 26, 5, 7, 85, 11),
    (12, 'AS Roma', 12, 23, 8, 7, 76, 12),
    (13, 'Tottenham Hotspur', 13, 21, 10, 7, 70, 13),
    (14, 'Arsenal', 14, 25, 6, 7, 82, 14)
    ]
)

# ---------------------- STADIUMS DATABASE ----------------------
setup_database(
    "stadiums.db",
    "STADIUMS",
    """
    CREATE TABLE IF NOT EXISTS STADIUMS(
        stadium_id INTEGER PRIMARY KEY,
        name VARCHAR(100),
        city VARCHAR(50),
        capacity INT,
        weather_impact VARCHAR(20)
    );
    """,
    [
    (1, 'DRV PNK Stadium', 'Miami', 18000, 'Low'),
    (2, 'Al-Awwal Park', 'Riyadh', 25000, 'High'),
    (3, 'King Fahd Stadium', 'Riyadh', 68000, 'Medium'),
    (4, 'Etihad Stadium', 'Manchester', 53400, 'Medium'),
    (5, 'Camp Nou', 'Barcelona', 99354, 'Low'),
    (6, 'Parc des Princes', 'Paris', 48712, 'Low'),
    (7, 'Old Trafford', 'Manchester', 74879, 'Medium'),
    (8, 'Anfield', 'Liverpool', 54074, 'High'),
    (9, 'Santiago Bernabeu', 'Madrid', 81044, 'Medium'),
    (10, 'Allianz Arena', 'Munich', 75024, 'Low'),
    (11, 'Wanda Metropolitano', 'Madrid', 68000, 'Low'),
    (12, 'Stadio Olimpico', 'Rome', 70000, 'High'),
    (13, 'Tottenham Hotspur Stadium', 'London', 62850, 'Low'),
    (14, 'Emirates Stadium', 'London', 60260, 'Low')
    ]
)

# ---------------------- FINANCIALS DATABASE ----------------------
setup_database(
    "financials.db",
    "FINANCIALS",
    """
    CREATE TABLE IF NOT EXISTS FINANCIALS(
        finance_id INTEGER PRIMARY KEY,
        team_id INTEGER,
        player_salary DECIMAL,
        transfer_fee DECIMAL,
        club_revenue DECIMAL,
        FOREIGN KEY (team_id) REFERENCES TEAMS(team_id)
    );
    """,
    [
    (1, 1, 50000000.00, 60000000.00, 350000000.00),
    (2, 2, 45000000.00, 55000000.00, 320000000.00),
    (3, 3, 48000000.00, 58000000.00, 340000000.00),
    (4, 4, 85000000.00, 120000000.00, 750000000.00),
    (5, 5, 87000000.00, 125000000.00, 850000000.00),
    (6, 6, 110000000.00, 140000000.00, 980000000.00),
    (7, 7, 85000000.00, 110000000.00, 740000000.00),
    (8, 8, 78000000.00, 95000000.00, 680000000.00),
    (9, 9, 95000000.00, 130000000.00, 900000000.00),
    (10, 10, 92000000.00, 125000000.00, 880000000.00),
    (11, 11, 75000000.00, 98000000.00, 700000000.00),
    (12, 12, 58000000.00, 74000000.00, 510000000.00),
    (13, 13, 65000000.00, 80000000.00, 560000000.00),
    (14, 14, 72000000.00, 87000000.00, 620000000.00)
    ]
)

# ---------------------- HOMES DATABASE ----------------------
setup_database(
    "homes.db",
    "HOMES",
    """
    CREATE TABLE IF NOT EXISTS HOMES(
        home_id INTEGER PRIMARY KEY,
        owner_name VARCHAR(50), 
        address VARCHAR(100), 
        city VARCHAR(50),
        state VARCHAR(50),
        zip_code VARCHAR(10),
        num_bedrooms INT, 
        num_bathrooms INT,
        square_footage INT,
        property_value DECIMAL
    );
    """,
    [
    (1, 'John Doe', '112 Oak St', 'Riverside', 'IL', '60661', 2, 1, 4353, 283448.81),
    (2, 'Jane Smith', '557 Maple Ave', 'Maplewood', 'IL', '87142', 3, 3, 942, 542915.97),
    (3, 'Alice Brown', '192 Pine St', 'Greenville', 'TX', '48595', 3, 1, 2946, 354330.15),
    (4, 'Michael Johnson', '986 Main St', 'Springfield', 'TX', '52796', 4, 3, 2303, 397724.79),
    (5, 'Emily Davis', '605 Willow Dr', 'Riverside', 'NY', '91139', 2, 2, 2228, 180181.38),
    (6, 'David Wilson', '113 Oak St', 'Maplewood', 'IL', '59930', 2, 3, 3530, 189720.76),
    (7, 'Sarah Miller', '255 Cedar Rd', 'Riverside', 'CA', '81400', 4, 3, 3109, 244984.35),
    (8, 'James Anderson', '459 Pine St', 'Lakeside', 'NY', '33998', 3, 4, 2076, 480253.69),
    (9, 'Emma Martinez', '682 Oak St', 'Greenville', 'IL', '61334', 4, 4, 4876, 255200.87),
    (10, 'Robert Thomas', '889 Willow Dr', 'Springfield', 'TX', '61371', 4, 3, 1724, 522667.83),
    (11, 'Olivia Hernandez', '287 Cedar Rd', 'Riverside', 'TX', '47390', 3, 2, 2099, 277705.47),
    (12, 'William Moore', '708 Birch Ln', 'Lakeside', 'FL', '30578', 5, 4, 2743, 257372.4),
    (13, 'Sophia Jackson', '235 Maple Ave', 'Springfield', 'CA', '56340', 3, 1, 3811, 512975.53),
    (14, 'Benjamin White', '958 Maple Ave', 'Riverside', 'CA', '90443', 2, 4, 2230, 253472.69),
    (15, 'Charlotte Harris', '356 Birch Ln', 'Springfield', 'NY', '59791', 4, 4, 2437, 195854.07),
    (16, 'Lucas Martin', '639 Willow Dr', 'Greenville', 'TX', '51821', 3, 2, 2683, 562349.01),
    (17, 'Amelia Thompson', '611 Cedar Rd', 'Springfield', 'CA', '43350', 1, 1, 1789, 182447.41),
    (18, 'Henry Garcia', '896 Main St', 'Riverside', 'NY', '64635', 1, 1, 908, 577877.4),
    (19, 'Mia Robinson', '137 Willow Dr', 'Maplewood', 'IL', '72537', 5, 1, 4668, 238506.95),
    (20, 'Alexander Clark', '743 Willow Dr', 'Greenville', 'CA', '26609', 5, 1, 4904, 486571.29),
    (21, 'Harper Lewis', '152 Maple Ave', 'Greenville', 'FL', '56761', 2, 1, 3311, 458751.07),
    (22, 'Daniel Walker', '717 Main St', 'Springfield', 'IL', '47699', 2, 3, 3463, 355054.2),
    (23, 'Evelyn Young', '464 Main St', 'Greenville', 'FL', '80681', 3, 1, 2152, 155616.53),
    (24, 'Matthew Hall', '428 Main St', 'Riverside', 'NY', '60226', 5, 2, 2864, 176687.47),
    (25, 'Abigail Allen', '621 Cedar Rd', 'Springfield', 'IL', '81944', 3, 3, 1457, 451198.87),
    (26, 'Joseph King', '626 Willow Dr', 'Lakeside', 'FL', '65022', 4, 2, 4713, 321132.83),
    (27, 'Ella Wright', '125 Birch Ln', 'Maplewood', 'TX', '56890', 3, 1, 2362, 297080.55),
    (28, 'Samuel Scott', '862 Cedar Rd', 'Springfield', 'TX', '56940', 5, 1, 3235, 379074.12),
    (29, 'Avery Green', '895 Maple Ave', 'Greenville', 'IL', '59309', 1, 4, 4533, 464872.7),
    (30, 'Jack Baker', '578 Pine St', 'Riverside', 'TX', '19901', 2, 3, 3537, 164609.75),
    (31, 'Scarlett Adams', '951 Cedar Rd', 'Greenville', 'IL', '89208', 5, 2, 3260, 586036.55),
    (32, 'Liam Nelson', '978 Main St', 'Maplewood', 'NY', '91667', 3, 2, 977, 385112.18),
    (33, 'Victoria Carter', '144 Oak St', 'Springfield', 'FL', '45432', 3, 4, 2972, 434418.19),
    (34, 'Mason Mitchell', '391 Oak St', 'Springfield', 'TX', '21675', 3, 4, 1828, 167891.31),
    (35, 'Luna Perez', '457 Oak St', 'Maplewood', 'FL', '48719', 3, 3, 4807, 549037.03),
    (36, 'Noah Roberts', '691 Willow Dr', 'Greenville', 'CA', '89565', 4, 1, 4666, 273828.31),
    (37, 'Grace Evans', '275 Willow Dr', 'Springfield', 'NY', '42457', 3, 2, 2808, 338775.25),
    (38, 'Elijah Edwards', '523 Pine St', 'Lakeside', 'FL', '31483', 5, 4, 2743, 373784.01),
    (39, 'Chloe Collins', '954 Pine St', 'Maplewood', 'IL', '41268', 1, 2, 1094, 565640.69),
    (40, 'Sebastian Stewart', '783 Cedar Rd', 'Greenville', 'NY', '97331', 4, 4, 1022, 204073.35),
    (41, 'Zoe Morris', '399 Cedar Rd', 'Springfield', 'CA', '98040', 2, 1, 4850, 295042.4),
    (42, 'Jackson Rogers', '513 Willow Dr', 'Maplewood', 'FL', '47736', 2, 4, 1279, 283250.43),
    (43, 'Penelope Reed', '813 Main St', 'Greenville', 'FL', '54190', 5, 4, 1520, 507157.62),
    (44, 'Logan Cook', '769 Main St', 'Springfield', 'TX', '72206', 2, 2, 3762, 511063.67),
    (45, 'Lily Bell', '856 Birch Ln', 'Lakeside', 'IL', '46531', 3, 3, 1133, 204939.73),
    (46, 'Nathan Murphy', '248 Main St', 'Springfield', 'TX', '19007', 5, 1, 3273, 572157.43),
    (47, 'Aria Bailey', '864 Birch Ln', 'Maplewood', 'NY', '88479', 5, 3, 3326, 268775.78),
    (48, 'Owen Cooper', '155 Pine St', 'Greenville', 'FL', '22422', 1, 1, 3659, 370737.16),
    (49, 'Hazel Richardson', '133 Willow Dr', 'Springfield', 'TX', '49137', 2, 1, 1341, 329040.63),
    (50, 'Wyatt Brooks', '362 Birch Ln', 'Springfield', 'IL', '20494', 1, 4, 1627, 437925.71)
    ]
)

# ---------------------- WATER DATABASE ----------------------
setup_database(
    "water.db",
    "WATER_SUPPLY",
    """
    CREATE TABLE IF NOT EXISTS WATER_SUPPLY(
        supply_id INTEGER PRIMARY KEY,
        home_id INTEGER,
        water_source VARCHAR(50),
        daily_usage_gallons DECIMAL,
        last_bill_date DATE,
        amount_due DECIMAL,
        FOREIGN KEY (home_id) REFERENCES HOMES(home_id)
    );
    """,
    [
    (1, 1, 'Reservoir', 860.19, '2024-04-24', 391.29),
    (2, 2, 'Groundwater', 767.06, '2024-06-25', 336.26),
    (3, 3, 'Lake', 284.78, '2024-11-18', 215.92),
    (4, 4, 'Municipal Supply', 736.82, '2024-04-23', 121.59),
    (5, 5, 'Municipal Supply', 884.16, '2024-02-02', 168.40),
    (6, 6, 'Reservoir', 324.68, '2024-01-28', 124.53),
    (7, 7, 'Groundwater', 865.86, '2024-04-24', 334.19),
    (8, 8, 'Municipal Supply', 996.53, '2024-08-06', 206.95),
    (9, 9, 'Groundwater', 542.57, '2024-06-03', 113.33),
    (10, 10, 'Lake', 226.21, '2024-02-09', 326.24),
    (11, 11, 'Municipal Supply', 338.56, '2024-01-11', 151.73),
    (12, 12, 'Lake', 597.94, '2024-03-03', 393.44),
    (13, 13, 'Groundwater', 783.53, '2024-02-15', 273.42),
    (14, 14, 'River', 472.42, '2024-05-07', 165.85),
    (15, 15, 'Reservoir', 764.72, '2024-02-11', 217.20),
    (16, 16, 'Reservoir', 831.27, '2024-05-06', 237.51),
    (17, 17, 'Municipal Supply', 129.09, '2024-02-06', 371.07),
    (18, 18, 'Groundwater', 978.53, '2024-12-04', 298.07),
    (19, 19, 'Groundwater', 172.36, '2024-11-19', 288.94),
    (20, 20, 'Groundwater', 842.91, '2024-10-15', 311.93),
    (21, 21, 'Municipal Supply', 628.45, '2024-12-05', 274.99),
    (22, 22, 'Lake', 176.80, '2024-06-24', 163.54),
    (23, 23, 'Reservoir', 988.45, '2024-04-06', 466.13),
    (24, 24, 'Lake', 205.29, '2024-01-28', 377.95),
    (25, 25, 'Lake', 811.00, '2024-12-12', 181.08),
    (26, 26, 'Reservoir', 702.36, '2024-08-24', 138.68),
    (27, 27, 'Groundwater', 671.50, '2024-12-17', 335.45),
    (28, 28, 'Reservoir', 995.22, '2024-07-10', 316.10),
    (29, 29, 'Lake', 673.55, '2024-06-22', 246.30),
    (30, 30, 'Groundwater', 335.00, '2024-10-16', 220.77),
    (31, 31, 'Reservoir', 498.52, '2024-12-26', 205.15),
    (32, 32, 'Lake', 671.18, '2024-09-11', 309.08),
    (33, 33, 'Reservoir', 717.80, '2024-06-09', 420.97),
    (34, 34, 'Lake', 616.33, '2024-11-18', 477.11),
    (35, 35, 'Municipal Supply', 315.07, '2024-01-20', 473.87),
    (36, 36, 'Municipal Supply', 172.08, '2024-04-08', 240.33),
    (37, 37, 'Reservoir', 323.80, '2024-01-14', 49.14),
    (38, 38, 'Lake', 553.89, '2024-02-27', 246.90),
    (39, 39, 'River', 936.20, '2024-10-25', 88.94),
    (40, 40, 'Lake', 459.48, '2024-06-20', 404.89),
    (41, 41, 'Groundwater', 839.92, '2024-12-21', 370.06),
    (42, 42, 'Groundwater', 123.00, '2024-04-05', 248.15),
    (43, 43, 'Municipal Supply', 440.11, '2024-05-11', 133.91),
    (44, 44, 'Reservoir', 118.78, '2024-09-07', 461.14),
    (45, 45, 'Lake', 234.49, '2024-05-09', 430.03),
    (46, 46, 'Groundwater', 809.92, '2024-11-16', 66.08),
    (47, 47, 'Lake', 366.16, '2024-03-05', 335.84),
    (48, 48, 'Municipal Supply', 549.30, '2024-05-15', 101.50),
    (49, 49, 'River', 940.16, '2024-09-06', 277.59),
    (50, 50, 'Reservoir', 390.04, '2024-02-06', 78.02)
    ]
)

# ---------------------- LAND DATABASE ----------------------
setup_database(
    "land.db",
    "LAND_PLOTS",
    """
    CREATE TABLE IF NOT EXISTS LAND_PLOTS(
        plot_id INTEGER PRIMARY KEY,
        home_id INTEGER,
        area_sqft DECIMAL,
        land_use_type VARCHAR(50),
        zoning_code VARCHAR(20),
        property_value DECIMAL,
        FOREIGN KEY (home_id) REFERENCES HOMES(home_id)
    );
    """,
    [
    (1, 1, 21568, 'Agricultural', 'C2', 228970.75),
    (2, 2, 20123, 'Commercial', 'C1', 180375.15),
    (3, 3, 26021, 'Residential', 'MU', 584663.26),
    (4, 4, 13360, 'Mixed-Use', 'R1', 364114.51),
    (5, 5, 41254, 'Residential', 'R1', 532374.38),
    (6, 6, 30517, 'Industrial', 'I1', 272505.18),
    (7, 7, 49948, 'Agricultural', 'A1', 472601.36),
    (8, 8, 28215, 'Commercial', 'C1', 405115.62),
    (9, 9, 35844, 'Mixed-Use', 'MU', 545269.30),
    (10, 10, 41893, 'Residential', 'R2', 526873.91),
    (11, 11, 15197, 'Agricultural', 'A1', 369590.02),
    (12, 12, 44894, 'Commercial', 'C2', 507343.61),
    (13, 13, 24317, 'Residential', 'R1', 389477.21),
    (14, 14, 27006, 'Industrial', 'I1', 163497.10),
    (15, 15, 31245, 'Mixed-Use', 'MU', 453111.83),
    (16, 16, 12469, 'Commercial', 'C2', 489973.54),
    (17, 17, 35288, 'Residential', 'R1', 364993.87),
    (18, 18, 15688, 'Agricultural', 'A1', 237570.94),
    (19, 19, 31979, 'Industrial', 'I1', 594194.26),
    (20, 20, 43396, 'Residential', 'R2', 568140.35),
    (21, 21, 18453, 'Mixed-Use', 'MU', 387493.11),
    (22, 22, 25673, 'Commercial', 'C1', 194250.42),
    (23, 23, 37641, 'Residential', 'R1', 475837.57),
    (24, 24, 41029, 'Agricultural', 'A1', 157462.90),
    (25, 25, 29915, 'Industrial', 'I1', 253165.37),
    (26, 26, 22447, 'Residential', 'R2', 593839.48),
    (27, 27, 44685, 'Commercial', 'C2', 482751.52),
    (28, 28, 39788, 'Mixed-Use', 'MU', 368116.98),
    (29, 29, 18906, 'Agricultural', 'A1', 193872.73),
    (30, 30, 29455, 'Industrial', 'I1', 512897.39),
    (31, 31, 41872, 'Residential', 'R1', 259483.64),
    (32, 32, 31684, 'Commercial', 'C1', 431207.91),
    (33, 33, 24591, 'Mixed-Use', 'MU', 275691.32),
    (34, 34, 43987, 'Agricultural', 'A1', 582634.45),
    (35, 35, 35163, 'Industrial', 'I1', 563749.30),
    (36, 36, 29488, 'Residential', 'R2', 327890.28),
    (37, 37, 36199, 'Commercial', 'C2', 496384.01),
    (38, 38, 31745, 'Mixed-Use', 'MU', 302014.85),
    (39, 39, 23678, 'Agricultural', 'A1', 159745.17),
    (40, 40, 27854, 'Industrial', 'I1', 421632.40),
    (41, 41, 14983, 'Residential', 'R1', 594320.19),
    (42, 42, 18476, 'Commercial', 'C1', 540262.84),
    (43, 43, 21735, 'Mixed-Use', 'MU', 384729.91),
    (44, 44, 30946, 'Agricultural', 'A1', 196251.67),
    (45, 45, 45129, 'Industrial', 'I1', 276491.38),
    (46, 46, 12369, 'Residential', 'R2', 360918.12),
    (47, 47, 39682, 'Commercial', 'C2', 512840.95),
    (48, 48, 17456, 'Mixed-Use', 'MU', 395187.56),
    (49, 49, 24971, 'Agricultural', 'A1', 326780.44),
    (50, 50, 36218, 'Industrial', 'I1', 458934.11)
    ]
)

# ---------------------- UTILITIES DATABASE ----------------------
setup_database(
    "utilities.db",
    "UTILITIES",
    """
    CREATE TABLE IF NOT EXISTS UTILITIES(
        utility_id INTEGER PRIMARY KEY,
        home_id INTEGER,
        electricity_provider VARCHAR(50),
        electricity_bill DECIMAL,
        gas_provider VARCHAR(50),
        gas_bill DECIMAL,
        last_bill_date DATE,
        FOREIGN KEY (home_id) REFERENCES HOMES(home_id)
    );
    """,
    [
    (1, 1, 'Green Energy', 141.66, 'Eco Gas', 189.91, '2024-09-14'),
    (2, 2, 'Green Energy', 208.74, 'Energy Pipeline', 91.78, '2024-03-03'),
    (3, 3, 'Green Energy', 267.86, 'Energy Pipeline', 187.04, '2024-01-26'),
    (4, 4, 'Green Energy', 52.48, 'Energy Pipeline', 184.07, '2024-09-06'),
    (5, 5, 'National Grid', 82.57, 'Eco Gas', 187.93, '2024-05-10'),
    (6, 6, 'Bright Light Electric', 74.96, 'National Gas', 68.53, '2024-04-28'),
    (7, 7, 'Green Energy', 86.43, 'Eco Gas', 114.29, '2024-06-21'),
    (8, 8, 'Springfield Power', 150.85, 'Energy Pipeline', 133.68, '2024-02-09'),
    (9, 9, 'Bright Light Electric', 90.79, 'National Gas', 70.29, '2024-08-15'),
    (10, 10, 'National Grid', 120.34, 'City Gas', 165.17, '2024-01-17'),
    (11, 11, 'Springfield Power', 200.85, 'Eco Gas', 91.46, '2024-10-22'),
    (12, 12, 'National Grid', 68.23, 'Energy Pipeline', 110.42, '2024-05-11'),
    (13, 13, 'Green Energy', 133.52, 'National Gas', 95.77, '2024-06-14'),
    (14, 14, 'Bright Light Electric', 108.29, 'Eco Gas', 141.84, '2024-11-28'),
    (15, 15, 'National Grid', 245.17, 'City Gas', 150.27, '2024-02-14'),
    (16, 16, 'Springfield Power', 195.32, 'Energy Pipeline', 117.38, '2024-09-02'),
    (17, 17, 'Bright Light Electric', 93.89, 'City Gas', 63.29, '2024-04-04'),
    (18, 18, 'Springfield Power', 272.56, 'Eco Gas', 135.77, '2024-07-19'),
    (19, 19, 'Green Energy', 127.90, 'National Gas', 159.32, '2024-12-05'),
    (20, 20, 'National Grid', 89.47, 'City Gas', 98.41, '2024-08-27'),
    (21, 21, 'Springfield Power', 163.74, 'Energy Pipeline', 187.53, '2024-01-30'),
    (22, 22, 'Bright Light Electric', 67.42, 'Eco Gas', 102.84, '2024-03-15'),
    (23, 23, 'National Grid', 110.92, 'National Gas', 76.54, '2024-02-23'),
    (24, 24, 'Green Energy', 146.68, 'Energy Pipeline', 99.77, '2024-06-10'),
    (25, 25, 'Springfield Power', 259.87, 'City Gas', 94.65, '2024-09-25'),
    (26, 26, 'Bright Light Electric', 202.43, 'Eco Gas', 173.29, '2024-10-31'),
    (27, 27, 'National Grid', 56.82, 'National Gas', 50.47, '2024-05-04'),
    (28, 28, 'Green Energy', 230.57, 'City Gas', 184.09, '2024-12-09'),
    (29, 29, 'Bright Light Electric', 187.65, 'Eco Gas', 90.68, '2024-11-12'),
    (30, 30, 'Springfield Power', 98.41, 'National Gas', 136.24, '2024-07-22'),
    (31, 31, 'Bright Light Electric', 76.23, 'City Gas', 122.85, '2024-06-08'),
    (32, 32, 'National Grid', 140.52, 'Eco Gas', 55.98, '2024-03-28'),
    (33, 33, 'Green Energy', 266.71, 'National Gas', 177.64, '2024-02-18'),
    (34, 34, 'Springfield Power', 85.99, 'City Gas', 112.77, '2024-10-05'),
    (35, 35, 'Bright Light Electric', 128.34, 'Eco Gas', 107.53, '2024-05-26'),
    (36, 36, 'National Grid', 220.68, 'National Gas', 85.39, '2024-04-16'),
    (37, 37, 'Green Energy', 77.83, 'City Gas', 189.67, '2024-08-29'),
    (38, 38, 'Springfield Power', 96.45, 'Energy Pipeline', 101.92, '2024-01-11'),
    (39, 39, 'Bright Light Electric', 155.29, 'Eco Gas', 128.17, '2024-09-09'),
    (40, 40, 'Green Energy', 204.11, 'National Gas', 145.67, '2024-11-20'),
    (41, 41, 'National Grid', 137.86, 'City Gas', 163.99, '2024-03-07'),
    (42, 42, 'Bright Light Electric', 152.49, 'Eco Gas', 187.42, '2024-07-30'),
    (43, 43, 'Springfield Power', 264.72, 'National Gas', 108.26, '2024-05-18'),
    (44, 44, 'Green Energy', 61.47, 'Energy Pipeline', 77.15, '2024-02-27'),
    (45, 45, 'Bright Light Electric', 171.35, 'City Gas', 95.32, '2024-04-21'),
    (46, 46, 'National Grid', 184.67, 'National Gas', 74.89, '2024-06-01'),
    (47, 47, 'Green Energy', 140.98, 'Energy Pipeline', 113.45, '2024-12-14'),
    (48, 48, 'Springfield Power', 75.86, 'Eco Gas', 68.92, '2024-08-07'),
    (49, 49, 'Bright Light Electric', 199.25, 'City Gas', 102.58, '2024-03-19'),
    (50, 50, 'Green Energy', 132.76, 'National Gas', 91.84, '2024-11-02')
]
)

# Setup Student Database
setup_database(
    "student.db",
    "STUDENT",
    """
    CREATE TABLE IF NOT EXISTS STUDENT(
        NAME VARCHAR(25), 
        CLASS VARCHAR(25), 
        AGE INT, 
        MARKS INT
    );
    """,
    [
        ('avsai', 'AI', 24, 25), ('vy', 'ML', 23, 24),
        ('virat', 'DL', 22, 33), ('raina', 'AI', 24, 31),
        ('john', 'ML', 22, 28), ('emma', 'DL', 21, 35),
        ('liam', 'AI', 26, 40), ('olivia', 'ML', 27, 38),
        ('noah', 'DL', 25, 29), ('sophia', 'AI', 22, 33),
        ('jackson', 'ML', 23, 30), ('ava', 'DL', 24, 27),
        ('aiden', 'AI', 28, 32), ('isabella', 'ML', 26, 31),
        ('lucas', 'DL', 22, 34), ('mia', 'AI', 29, 29),
        ('elijah', 'ML', 30, 36), ('amelia', 'DL', 21, 37),
        ('benjamin', 'AI', 24, 39), ('harper', 'ML', 25, 35),
        ('ethan', 'DL', 28, 31), ('evelyn', 'AI', 27, 30),
        ('mason', 'ML', 26, 28), ('abigail', 'DL', 24, 40),
        ('logan', 'AI', 23, 34), ('ella', 'ML', 22, 37),
        ('alexander', 'DL', 30, 32), ('scarlett', 'AI', 25, 36),
        ('james', 'ML', 27, 33), ('chloe', 'DL', 29, 38)
    ]
)

# Setup Employee Database
setup_database(
    "employees.db",
    "EMPLOYEES",
    """
    CREATE TABLE IF NOT EXISTS EMPLOYEES(
        NAME VARCHAR(25), 
        DEPARTMENT VARCHAR(25), 
        SALARY INT, 
        EXPERIENCE INT
    );
    """,
    [
        ('John', 'IT', 70000, 5), ('Jane', 'HR', 65000, 6),
        ('Bob', 'Finance', 80000, 8), ('Alice', 'IT', 75000, 7),
        ('David', 'Marketing', 72000, 4), ('Emma', 'HR', 68000, 7),
        ('Michael', 'IT', 85000, 10), ('Sophia', 'Finance', 90000, 12),
        ('Daniel', 'Sales', 65000, 5), ('Olivia', 'Marketing', 78000, 8),
        ('James', 'IT', 87000, 9), ('Charlotte', 'HR', 70000, 6),
        ('William', 'Finance', 92000, 13), ('Amelia', 'Sales', 69000, 4),
        ('Ethan', 'IT', 83000, 10), ('Isabella', 'Marketing', 74000, 5),
        ('Mason', 'HR', 66000, 6), ('Mia', 'Finance', 89000, 11),
        ('Alexander', 'Sales', 72000, 7), ('Harper', 'IT', 79000, 8),
        ('Benjamin', 'Marketing', 81000, 9), ('Evelyn', 'HR', 75000, 7),
        ('Henry', 'Finance', 91000, 14), ('Abigail', 'Sales', 68000, 5),
        ('Liam', 'IT', 86000, 9), ('Ella', 'Marketing', 77000, 6),
        ('Lucas', 'HR', 73000, 7), ('Scarlett', 'Finance', 88000, 12),
        ('Daniel', 'Sales', 70000, 6), ('Victoria', 'IT', 80000, 10)
    ]
)

# Setup Sales Database
setup_database(
    "sales.db",
    "SALES",
    """
    CREATE TABLE IF NOT EXISTS SALES(
        PRODUCT VARCHAR(25), 
        REGION VARCHAR(25), 
        SALES INT, 
        REVENUE INT
    );
    """,
    [
        ('Laptop', 'North', 50, 100000), ('Phone', 'South', 100, 200000),
        ('Tablet', 'East', 30, 60000), ('Monitor', 'West', 20, 40000),
        ('Desktop', 'North', 40, 80000), ('Smartwatch', 'South', 90, 150000),
        ('Headphones', 'East', 70, 70000), ('Speakers', 'West', 25, 50000),
        ('Keyboard', 'North', 80, 160000), ('Mouse', 'South', 120, 180000),
        ('Camera', 'East', 45, 85000), ('Printer', 'West', 35, 75000),
        ('Smart TV', 'North', 55, 110000), ('Gaming Console', 'South', 95, 250000),
        ('Router', 'East', 60, 90000), ('Power Bank', 'West', 30, 60000),
        ('Monitor', 'North', 20, 50000), ('Projector', 'South', 75, 140000),
        ('SSD', 'East', 100, 220000), ('Hard Drive', 'West', 90, 180000),
        ('Graphics Card', 'North', 15, 300000), ('Motherboard', 'South', 35, 85000),
        ('RAM', 'East', 25, 65000), ('Processor', 'West', 55, 275000),
        ('VR Headset', 'North', 12, 48000), ('Fitness Band', 'South', 85, 90000),
        ('Bluetooth Speaker', 'East', 50, 95000), ('Smart Glasses', 'West', 22, 110000),
        ('Drone', 'North', 18, 200000), ('E-Reader', 'South', 65, 120000)
    ]
)
