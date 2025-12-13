# Python Codes Accompanying Manuscript  
**“Climate-driven impacts in renewable resources alter green hydrogen supply”**

This repository contains the Python codes accompanying a manuscript that systematically assesses how climate change affects the cost of renewable-powered green hydrogen production at the global scale. The codes reproduce the core analytical results presented in the paper, including climate-driven cost changes, uncertainty analysis, and spatially explicit mapping of hydrogen production costs.

---

## Scope of the Analysis

The codebase supports the following core tasks:

- Processing multi-model CMIP6 climate projections for wind, solar radiation, and temperature  
- Deriving physically interpretable climate features relevant to renewable electricity generation  
- Sampling future technology cost uncertainty using Monte Carlo and Latin Hypercube methods  
- Predicting grid-level hydrogen costs using machine-learning surrogate models  

The analysis focuses on renewable-based electrolysis systems, explicitly capturing interactions between wind, solar, storage, and electrolyzer technologies.

---




## Dependency Management

Python dependencies are managed using standard scientific Python packages (e.g., `numpy`, `pandas`, `xarray`, `scipy`, `xgboost`, `lightgbm`, `dask`, `geopandas`).  
It is recommended to run the code within a dedicated virtual environment (e.g., `conda` or `venv`) to ensure reproducibility.

---

## Citation

If you use this code or build upon it, please cite the accompanying manuscript:

J. Ren, Q. Zhang, Sl. Zhang, Sh. Zhang, W. Chen (2025). Climate-driven impacts in renewable resources alter green hydrogen supply. Working paper.

A BibTeX entry for LaTeX users is
 ```latex
@ARTICLE{,
  title = {Climate-driven impacts in renewable resources alter green hydrogen supply.},
  author = {J. Ren, Q. Zhang, Sl. Zhang, Sh. Zhang, W. Chen},
  year = {2025},
  note = {Working paper},
}
```
---

## License

See file `LICENSE` or navigate to https://www.gnu.org/licenses/gpl-3.0.html.

## Interactive Web Visualization

An interactive, project-level map for exploring the climate-driven impacts on green hydrogen costs is available online:

👉 **[Open the interactive HTML map](https://onismyh.github.io/climate-impacts-hydrogen)**

If the link above does not open, try the GitHub Pages site root:

🔗 https://onismyh.github.io/climate-impacts-hydrogen/