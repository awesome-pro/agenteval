import agenteval
from agenteval import Tracer


async def search_orders(order_id: str) -> str:
    return f"Order {order_id} was delivered yesterday."


async def customer_support_agent(message: str, search_orders) -> str:
    order_info = await search_orders(order_id="123")
    return f"I checked your order. {order_info}"


@agenteval.test(n=5, threshold=0.8)
async def test_agent_checks_order_status(tracer: Tracer) -> None:
    search = tracer.wrap(search_orders)

    async with tracer.run(input="Where is my order?") as run:
        result = await customer_support_agent(
            "Where is my order?",
            search_orders=search,
        )
        run.set_output(result)

    (
        tracer.assert_that()
        .called_tool("search_orders")
        .completed_within_steps(3)
        .response_contains("delivered", case_sensitive=False)
        .no_errors()
        .check()
    )
