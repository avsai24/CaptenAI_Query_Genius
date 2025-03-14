schema = {
  "database_folder": "/Users/venkatasaiancha/Documents/all_concepts/multi_databse_retriver/sqlite_databases",
  "databases": [
    {
      "database_name": "players.db",
      "tables": [
        {
          "table_name": "PLAYERS",
          "columns": [
            {"name": "player_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "name", "type": "VARCHAR(100)"},
            {"name": "team_id", "type": "INTEGER"},
            {"name": "position", "type": "VARCHAR(50)"},
            {"name": "goals", "type": "INT"},
            {"name": "assists", "type": "INT"},
            {"name": "matches", "type": "INT"}
          ],
          "foreign_keys": [
            {"column": "team_id", "references": "TEAMS(team_id)"}
          ]
        }
      ]
    },
    {
      "database_name": "teams.db",
      "tables": [
        {
          "table_name": "TEAMS",
          "columns": [
            {"name": "team_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "team_name", "type": "VARCHAR(50)"},
            {"name": "stadium_id", "type": "INTEGER"},
            {"name": "wins", "type": "INT"},
            {"name": "losses", "type": "INT"},
            {"name": "draws", "type": "INT"},
            {"name": "points", "type": "INT"},
            {"name": "finance_id", "type": "INTEGER"}
          ],
          "foreign_keys": [
            {"column": "stadium_id", "references": "STADIUMS(stadium_id)"},
            {"column": "finance_id", "references": "FINANCIALS(finance_id)"}
          ]
        }
      ]
    },
    {
      "database_name": "stadiums.db",
      "tables": [
        {
          "table_name": "STADIUMS",
          "columns": [
            {"name": "stadium_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "name", "type": "VARCHAR(100)"},
            {"name": "city", "type": "VARCHAR(50)"},
            {"name": "capacity", "type": "INT"},
            {"name": "weather_impact", "type": "VARCHAR(20)"}
          ]
        }
      ]
    },
    {
      "database_name": "financials.db",
      "tables": [
        {
          "table_name": "FINANCIALS",
          "columns": [
            {"name": "finance_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "team_id", "type": "INTEGER"},
            {"name": "player_salary", "type": "DECIMAL"},
            {"name": "transfer_fee", "type": "DECIMAL"},
            {"name": "club_revenue", "type": "DECIMAL"}
          ],
          "foreign_keys": [
            {"column": "team_id", "references": "TEAMS(team_id)"}
          ]
        }
      ]
    },
    {
      "database_name": "homes.db",
      "tables": [
        {
          "table_name": "HOMES",
          "columns": [
            {"name": "home_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "owner_name", "type": "VARCHAR(50)"},
            {"name": "address", "type": "VARCHAR(100)"},
            {"name": "city", "type": "VARCHAR(50)"},
            {"name": "state", "type": "VARCHAR(50)"},
            {"name": "zip_code", "type": "VARCHAR(10)"},
            {"name": "num_bedrooms", "type": "INT"},
            {"name": "num_bathrooms", "type": "INT"},
            {"name": "square_footage", "type": "INT"},
            {"name": "property_value", "type": "DECIMAL"}
          ]
        }
      ]
    },
    {
      "database_name": "water.db",
      "tables": [
        {
          "table_name": "WATER_SUPPLY",
          "columns": [
            {"name": "supply_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "home_id", "type": "INTEGER"},
            {"name": "water_source", "type": "VARCHAR(50)"},
            {"name": "daily_usage_gallons", "type": "DECIMAL"},
            {"name": "last_bill_date", "type": "DATE"},
            {"name": "amount_due", "type": "DECIMAL"}
          ],
          "foreign_keys": [
            {"column": "home_id", "references": "HOMES(home_id)"}
          ]
        }
      ]
    },
    {
      "database_name": "utilities.db",
      "tables": [
        {
          "table_name": "UTILITIES",
          "columns": [
            {"name": "utility_id", "type": "INTEGER PRIMARY KEY"},
            {"name": "home_id", "type": "INTEGER"},
            {"name": "electricity_provider", "type": "VARCHAR(50)"},
            {"name": "electricity_bill", "type": "DECIMAL"},
            {"name": "gas_provider", "type": "VARCHAR(50)"},
            {"name": "gas_bill", "type": "DECIMAL"},
            {"name": "last_bill_date", "type": "DATE"}
          ],
          "foreign_keys": [
            {"column": "home_id", "references": "HOMES(home_id)"}
          ]
        }
      ]
    },
    {
      "database_name": "student.db",
      "tables": [
        {
          "table_name": "STUDENT",
          "columns": [
            {"name": "NAME", "type": "VARCHAR(25)"},
            {"name": "CLASS", "type": "VARCHAR(25)"},
            {"name": "AGE", "type": "INT"},
            {"name": "MARKS", "type": "INT"}
          ]
        }
      ]
    },
    {
      "database_name": "employees.db",
      "tables": [
        {
          "table_name": "EMPLOYEES",
          "columns": [
            {"name": "NAME", "type": "VARCHAR(25)"},
            {"name": "DEPARTMENT", "type": "VARCHAR(25)"},
            {"name": "SALARY", "type": "INT"},
            {"name": "EXPERIENCE", "type": "INT"}
          ]
        }
      ]
    },
    {
      "database_name": "sales.db",
      "tables": [
        {
          "table_name": "SALES",
          "columns": [
            {"name": "PRODUCT", "type": "VARCHAR(25)"},
            {"name": "REGION", "type": "VARCHAR(25)"},
            {"name": "SALES", "type": "INT"},
            {"name": "REVENUE", "type": "INT"}
          ]
        }
      ]
    }
  ]
}