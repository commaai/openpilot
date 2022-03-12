{% for tier, categories, cars in tiers %}
## {{tier}} Cars

|{{categories | join('|')}}|
|---|---|---|:---:|:---:|:---:|:---:|:---:|
{% for car in cars %}
|{{car | join('|')}}|
{% endfor %}

{% endfor %}
