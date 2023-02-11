defmodule ElixirNeuralNetwork.MixProject do
  use Mix.Project

  def project do
    [
      app: :elixir_neural_network,
      version: "0.1.0",
      elixir: "~> 1.14",
      name: "Elixir Neurales Netzwerk",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ex_doc, "~> 0.21", only: :dev, runtime: false},
      {:axon, "~> 0.3.0"},
      {:exla, "~> 0.4.0"},
      {:nx, "~> 0.4.0", override: true},
      {:scidata, "~> 0.1.9"},
      {:table_rex, "~> 3.1.1"}
    ]
  end

  defp docs do
    [
      extras: ["README.md"],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.css" integrity="sha384-vKruj+a13U8yHIkAyGgK1J3ArTLzrFGBbBc0tDp4ad/EyewESeXE/Iv67Aj8gKZ0" crossorigin="anonymous">
    <script type="module">
      import renderMathInElement from "https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/contrib/auto-render.mjs";
      renderMathInElement(document.body, {
          // customised options
          // • auto-render specific keys, e.g.:
          delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false},
          ],
          // • rendering keys, e.g.:
          throwOnError : false
        });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
