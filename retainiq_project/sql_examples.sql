-- Churn rate by contract
SELECT contract,
       COUNT(*) AS customers,
       ROUND(AVG(churn) * 100, 2) AS churn_percent
FROM customers
GROUP BY contract
ORDER BY churn_percent DESC;

-- Monthly revenue currently carried by each risk segment
SELECT risk_level,
       COUNT(*) AS customer_count,
       ROUND(SUM(monthly_charges), 2) AS monthly_revenue
FROM customers
GROUP BY risk_level
ORDER BY monthly_revenue DESC;

-- Highest churn probability customers
SELECT customer_id, churn_probability, risk_level, contract,
       tenure_months, monthly_charges
FROM customers
ORDER BY churn_probability DESC
LIMIT 20;
