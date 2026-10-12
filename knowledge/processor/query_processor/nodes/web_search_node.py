import asyncio
import json

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebSearchNode(BaseNode):
    name = "web_search_node"
    def process(self, state: QueryGraphState) -> QueryGraphState:
        #1. 参数校验
        rewritten_query, item_names = self._validate_state(state)
        #2. 调用mcp工具进行网络检索
        execute_tool_result = asyncio.run(self.mcp_web_search(rewritten_query))
        #3. 对mcp检索的结果进行格式化
        #3.1 获取查询结果中的text，这其实就是一个json字符串
        result_text = execute_tool_result.content[0].text
        #3.2 对json字符串进行反序列化
        json_object = json.loads(result_text)
        #3.3 取出pages
        pages = json_object["pages"]

        #声明一个存储web_search结果的容器
        web_search_docs = []
        #3.4 遍历pages
        for page in pages:
            #3.4.1 获取snippet、title、url
            snippet = page["snippet"]
            title = page["title"]
            url = page["url"]
            web_search_docs.append({
                "snippet":snippet,
                "title":title,
                "url":url
            })

        return {"web_search_docs":web_search_docs}

    async def mcp_web_search(self,rewritten_query:str):
        # 百炼的 MCP 服务端只实现到 2024-11-05 旧协议：收到新协议的 server/discover 探测时
        # 返回 500 + 空 body，而不是 JSON-RPC 的 METHOD_NOT_FOUND。mcp 客户端的 mode="auto"
        # 会先发这个探测，且把握手阶段的 5xx 当成服务端故障直接断连（不回落旧协议），
        # 所以这里显式用 mode="legacy" 跳过探测，直接走 initialize 握手。
        # 1. 创建HTTP客户端（携带认证头）
        http_client = create_mcp_http_client(
            headers={"Authorization": self.config.openai_api_key},  # 认证头
        )
        async with http_client:
            # 2. 创建MCP客户端
            transport = streamable_http_client(
                self.config.mcp_dashscope_base_url,  # MCP 服务端点
                http_client=http_client,
                terminate_on_close=True,  # 关闭时终止连接
            )
            client = Client(transport, mode="legacy", cache=None)
            async with client as connected_client:
                #3. 调用mcp中的网络搜索的工具
                execute_tool_result = await connected_client.call_tool(
                    name="bailian_web_search",
                    arguments={"query": rewritten_query, "count": 3}
                )

                return execute_tool_result

    def _validate_state(self, state: QueryGraphState):
        # 1. 获取rewritten_query,item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        # 2. 校验rewritten_query,item_names
        if not rewritten_query or not isinstance(rewritten_query, str):
            self.logger.error("rewritten_query不能为空以及类型必须是str")
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            self.logger.error("item_names不能为空以及类型必须是list")
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

if __name__ == '__main__':
    # 1. 创建state
    state = {
        "rewritten_query": "RS PRO RS-12 数字万用表的使用方法是什么?",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }
    node = WebSearchNode()
    final_state = node.process(state)
    print(final_state)