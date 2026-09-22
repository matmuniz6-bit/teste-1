"""Memory-efficient Trading Strategy pair resolution helpers.

Avoids downloading the global pair universe. The initial simulator milestone
supports the canonical Ethereum Uniswap V3 WETH/USDC 5 bps pool only.
"""
from __future__ import annotations

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeType
from tradingstrategy.pair import DEXPair


SUPPORTED_CHAIN = ChainId.ethereum
SUPPORTED_EXCHANGE = "uniswap-v3"
SUPPORTED_BASE = "WETH"
SUPPORTED_QUOTE = "USDC"
SUPPORTED_FEE = 0.0005

_TOKEN_DECIMALS = {
    "WETH": 18,
    "USDC": 6,
}


def _normalise_chain_id(value) -> ChainId:
    if isinstance(value, ChainId):
        return value
    return ChainId(int(value))


def resolve_pair_lightweight(
    client,
    chain_id: ChainId = SUPPORTED_CHAIN,
    exchange_slug: str = SUPPORTED_EXCHANGE,
    base_token: str = SUPPORTED_BASE,
    quote_token: str = SUPPORTED_QUOTE,
    fee_tier: float = SUPPORTED_FEE,
) -> DEXPair:
    """Resolve one well-known pool through Trading Strategy's lightweight /top API."""
    chain_id = _normalise_chain_id(chain_id)
    base_token = base_token.upper()
    quote_token = quote_token.upper()

    if (
        chain_id != SUPPORTED_CHAIN
        or exchange_slug != SUPPORTED_EXCHANGE
        or base_token != SUPPORTED_BASE
        or quote_token != SUPPORTED_QUOTE
        or abs(float(fee_tier) - SUPPORTED_FEE) > 1e-12
    ):
        raise ValueError(
            "Initial native simulator supports only "
            "Ethereum / uniswap-v3 / WETH-USDC / 0.0005"
        )

    top_reply = client.fetch_top_pairs(
        chain_ids={chain_id},
        exchange_slugs={exchange_slug},
        limit=100,
    )

    candidates = list(top_reply.included) + list(top_reply.excluded)
    match = next(
        (
            p
            for p in candidates
            if p.chain_id == chain_id.value
            and p.exchange_slug == exchange_slug
            and p.base_token.upper() == base_token
            and p.quote_token.upper() == quote_token
            and abs(float(p.fee) - float(fee_tier)) <= 1e-12
        ),
        None,
    )
    if match is None:
        raise LookupError(
            f"Pool not returned by Trading Strategy /top: "
            f"{chain_id.name} {exchange_slug} {base_token}/{quote_token} {fee_tier}"
        )

    base_address = match.base_token_address.lower()
    quote_address = match.quote_token_address.lower()

    # Uniswap token0/token1 ordering is ascending by address.
    if int(base_address, 16) < int(quote_address, 16):
        token0_address = base_address
        token0_symbol = base_token
        token0_decimals = _TOKEN_DECIMALS[base_token]
        token1_address = quote_address
        token1_symbol = quote_token
        token1_decimals = _TOKEN_DECIMALS[quote_token]
    else:
        token0_address = quote_address
        token0_symbol = quote_token
        token0_decimals = _TOKEN_DECIMALS[quote_token]
        token1_address = base_address
        token1_symbol = base_token
        token1_decimals = _TOKEN_DECIMALS[base_token]

    return DEXPair(
        pair_id=int(match.pair_id),
        chain_id=chain_id,
        exchange_id=int(match.exchange_id),
        address=match.pool_address.lower(),
        token0_address=token0_address,
        token1_address=token1_address,
        token0_symbol=token0_symbol,
        token1_symbol=token1_symbol,
        dex_type=ExchangeType.uniswap_v3,
        base_token_symbol=base_token,
        quote_token_symbol=quote_token,
        token0_decimals=token0_decimals,
        token1_decimals=token1_decimals,
        exchange_slug=exchange_slug,
        pair_slug=f"{base_token.lower()}-{quote_token.lower()}",
        fee=int(round(float(fee_tier) * 10_000)),
    )


def resolve_pair_dataframe(client, **kwargs):
    """Return a one-row pair dataframe accepted directly by load_partial_data()."""
    pair = resolve_pair_lightweight(client, **kwargs)
    return pair, DEXPair.convert_to_dataframe([pair])
