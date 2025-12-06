import streamlit as st

def main():
    st.title(" Mapping Economic Bias: DML + Latent Space Mapping")
    st.subheader("Overview of Our STA 160 Project")

    st.markdown(
        """
This page explains how our framework works for analyzing
**economic bias in player salaries** using:

- **Double Machine Learning (DML)** to isolate the effect of bias factors  
- **Latent Space Mapping (LSM)** to turn those effects into a **visual map** of players and biases  
        """
    )

    st.markdown("---")

    # --- High-level overview section ---
    st.header("Big Picture: What Are We Trying To Do?")

    st.markdown(
        """
In sports like the NBA, salaries aren’t always determined solely by performance.
They can also be influenced by things like:

- **Draft position**
- **Age**
- **Team market size**
- **Owner wealth**
- **Contract rules** (rookie scale, max contracts, etc.)

Our goal is to answer questions like:

> **“What hidden factors are helping decide player salaries, and which players are affected
> by the same kinds of bias?”**
        """
    )

    st.markdown(
        """
Classic regression just gives us a table of coefficients. That can tell us:

- Whether a factor is statistically significant  
- Roughly how salary changes with that factor  

But it does **NOT** tell us:

- How **biases interact** with each other  
- Which **groups of players** are affected in similar ways  
- How **contract rules** (e.g., fixed rookie or max deals) distort the picture  
        """
    )

    st.info(
        "So instead of just a wall of coefficients, our framework builds a "
        "**map** of players and bias factors, showing how they’re connected."
    )

    st.markdown("---")

    # --- Tabs for the 3 main steps ---

    st.subheader("Fix the Salary Model (Stratified DML)")

    st.markdown(
            """
The first problem: **not all salaries are set by the free market.**

- **Rookies** are stuck on a fixed rookie scale.
- **Superstars** might be stuck at a **max contract**.
- Those salaries don’t move freely, even if performance skyrockets.

If we naively train a model on **everyone**, then:

- a high-performing rookie might look **“underpaid”**,  
- but that’s because of **CBA rules**, not market bias.

To avoid this, we use a **stratified Double Machine Learning** approach:
we **only learn the market relationship from truly negotiated contracts**.
            """
        )

    st.markdown("### What we actually do:")

    st.markdown(
            """
1. **Filter the data**
   - Keep only players on **free-market / negotiated contracts**.
   - Drop rookies and max-deal players during the *learning* phase.

2. **Predict salary from performance**
   - Train a flexible ML model (e.g. gradient boosting)  
     to predict salary from performance stats \\(X\\).
   - The **residual** here is the part of salary **not** explained by performance.

3. **Predict each bias factor from performance**
   - For each bias variable (e.g. age, draft number, market size),  
     train another model that predicts that factor from stats.
   - The residual is the part of the bias factor that is **independent of performance**.

4. **Regress residuals on residuals**
   - Regress “salary residual” on all the “bias residuals”.
   - The resulting coefficients estimate the **market price of each bias factor**
     after controlling for performance.
            """
        )

    st.success(
            "End of Step 1: We get a clean estimate of how much each bias factor "
            "is worth in the free market, plus residuals showing how unusual each "
            "player’s situation is relative to their performance."
        )


    st.markdown("---")
    st.header(" Why This Framework Is Different")

    st.markdown(
        """
Traditional regression answers:

> “Does this factor matter on average, and by how much?”

Our framework answers:

> “**Which** players are driven by **which** biases,  
> how **strong** those biases are for each person,  
> and how all of this looks in a **single visual map**.”

This is what makes the method useful for:

- General managers and front offices  
- Agents and players  
- Researchers studying discrimination and inefficiency in markets  
        """
    )


if __name__ == '__main__':
    main()