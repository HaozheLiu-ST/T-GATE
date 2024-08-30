import setuptools

setuptools.setup(
    name="tgate",
    version="v1.0.0",
    author="Wentian Zhang",
    author_email="zhangwentianml@gmail.com",
    description="TGATE-V2: Faster Diffusion Through Temporal Attention Decomposition.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HaozheLiu-ST/T-GATE/tree/releases/",
    package_dir={"": "src"},
    packages=setuptools.find_packages('src'),
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "diffusers>=0.29.0",
        "transformers",
        "DeepCache==0.1.1",
        "torch",
        "accelerate",
    ],
    python_requires='>=3.8.0',
)
