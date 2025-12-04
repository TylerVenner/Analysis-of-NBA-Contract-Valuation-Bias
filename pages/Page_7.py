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
    tab1, tab2, tab3 = st.tabs(
        [
            "Step 1: Fix the Salary Model (DML)",
            "Step 2: Build the Attribution Matrix",
            "Step 3: Turn It Into a Map",
        ]
    )

    # ------------------- TAB 1 -------------------
    with tab1:
        st.subheader(" Step 1 — Fix the Salary Model (Stratified DML)")

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

        st.markdown("### What we actually do in Step 1")

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

    # ------------------- TAB 2 -------------------
    with tab2:
        st.subheader(" Step 2 — Build the Attribution Matrix")

        st.markdown(
            """
Now we want to know, **for each player**, how much each bias factor actually matters.

From Step 1, we have:

-  **$\\hat{\\gamma}_j$** — the market price of bias factor *j*
- **$\\hat{\\epsilon}_{Z(j), i}$** — how unusual player *i* is on factor *j* after controlling for performance

We combine these into a matrix **L** where each entry $L_{ij}$ is:

> **How strongly does bias factor *j* influence player *i*’s salary, in absolute terms?**
            """
        )

        st.latex(r"L_{ij} = \big|\hat{\gamma}_j \cdot \hat{\epsilon}_{Z(j), i}\big|")

        st.markdown(
            """
Why this is useful:

- Multiplying the **price** of a bias by the **player’s residual** for that bias  
  converts everything into the **same unit**:  
  > “log-dollars of influence”

- Taking the **absolute value** means we care about **how important** the factor is,
  not whether it makes the player overpaid or underpaid.

So:

- **Large $L_{ij}$** → that bias is a **major driver** of unexplained salary for player *i*  
- **Small $L_{ij}$** → that bias barely matters for that player  
            """
        )

        with st.expander("💡 Intuition with examples"):
            st.markdown(
                """
- If a rookie was drafted way earlier than their stats suggest, and “draft number”
  is highly priced in the market, then their **draft attribution** will be big.

- If a veteran’s salary is well explained by performance and not much else,
  all their attribution scores may be small → they are close to **“pure performance”**.
                """
            )

        st.info(
            "End of Step 2: We now have a **personalized bias profile** for each player, "
            "telling us which factors matter most for them."
        )

    # ------------------- TAB 3 -------------------
    with tab3:
        st.subheader(" Step 3 — Turn It Into a Map (Latent Space Mapping)")

        st.markdown(
            """
Now we take the Attribution Matrix **L** and convert it into a **map**.

We want a picture where:

- **Bias factors** (like Draft, Age, Market Size) act like **anchors** in space.
- **Players** are points that are pulled toward the anchors that influence them the most.
- Players that are influenced by **similar combinations of biases** end up **close together**.
            """
        )

        st.markdown("### Conceptual modeling idea")

        st.markdown(
            """
- Treat each **player** and each **bias factor** as a point with some uncertainty
  (a Gaussian distribution in space).

- Define a similarity function where:
  - If a player is **close** to a factor, the predicted attribution is **high**.
  - If a player is **far**, the predicted attribution is **low**.

- Use math (a closed-form formula for expected similarity) to predict $\\hat{L}_{ij}$ 
for any configuration of player and factor positions.

- Then **optimize** the positions of all points so that the predicted $\\hat{L}_{ij}$
  matches the actual $L_{ij}$ as closely as possible.
            """
        )

        st.markdown("### What the final map shows")

        st.markdown(
            """
In the final interactive 3D plot:

- **Bias Anchors**:
  - Represent factors like “Age”, “Draft Number”, “Market Size”.
  - Anchors that are close to each other tend to influence the same players.

- **Player Points**:
  - Cluster around the anchors that most strongly affect their salaries.
  - A tight group around “Draft Number” suggests a group of players whose pay is
    tightly tied to draft status (often rookies or young stars).

- **The Center / Void**:
  - Players near the center have **low attribution across all bias factors**.
  - Their salaries are mostly explained by **performance alone**.
            """
        )


        st.success(
            "End of Step 3: We’ve turned raw coefficients and residuals into a "
            "**geometry of bias** — a map that shows which players share the same "
            "underlying economic forces."
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
