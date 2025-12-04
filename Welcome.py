import streamlit as st
import pandas as pd



# Set the page configuration
st.set_page_config(
    page_title="NBA Contract Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏀 Analysis of NBA Contract Valuation")
st.subheader("STA 160 Capstone Project - Group 5")


st.markdown("""
##  Research Question
How can we move beyond traditional sports analytics to reveal the hidden patterns that shape how athletes are valued? Our project asks a simple but powerful question:  
**What does economic bias *look like* when you map it, and what hidden forces shape the way NBA players are paid?**

While salary data is public, the forces behind those numbers—hype, age, draft narratives, market size—are rarely visible. We set out to uncover whether these influences form identifiable structures, clusters, or “bias neighborhoods” inside the NBA economy.

---

##  Context
Sports analysts and economists often rely on charts, tables, and regression outputs to study discrimination and market inefficiency. But these tools can only tell us *which* factors matter—not how those factors relate to each other, or how they collectively shape real player careers.

NBA salaries add an extra layer of difficulty: many players earn “fixed” rookie or max contracts that have nothing to do with free-market value. This makes it nearly impossible to understand true economic bias using standard models.

Our team wanted a better way to see the full picture—not just the numbers, but the underlying *structure*.

---

##  Discussion & Why This Matters
Our project introduces a new way to visualize how bias operates in professional sports. Instead of ending with a wall of coefficients, we build an interactive **Bias Map** that uncovers clusters of players affected by the same underlying forces—whether that’s age, hype, market size, or draft history.

Early results already show striking patterns: a clear “rookie cluster,” a defined “aging-star zone,” and even a central region where players appear to be valued almost entirely on performance. By turning complex analytics into an intuitive map, our work gives fans, analysts, and decision-makers a new lens for understanding fairness, value, and hidden inefficiencies in the NBA.

**We invite you to explore our preliminary visualizations below and see how economic bias takes shape when you give it a geometry.**
""")

""" 
            
### How to Use This App
Use the sidebar on the left to navigate between the different sections of our analysis:
- **Home:** You are here.
- **Data Overview:** (Example Page) A look at the raw data we collected.
- **Model Results:** (Example Page) The final output from our DML model.
- **[Teammate Pages]:** Explorations and specific analyses from each team member.

---
"""
