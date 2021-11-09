from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def send_metrics(metrics):
    registry = CollectorRegistry()
    for key, value in metrics.items():
        g = Gauge(f'{key}_mape', f'MAPE value for {key}', registry=registry)
        g.set(value)
    push_to_gateway('localhost:9091', job='pushgateway', registry=registry)
