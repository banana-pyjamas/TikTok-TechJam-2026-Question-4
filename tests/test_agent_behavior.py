from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class AgentBehaviorTest(unittest.TestCase):
    # Use a small fake catalog so the tests run quickly.
    # The real catalog has 50,000 products and is not needed for these tests.

    def setUp(self) -> None:
        # Create two products with clearly different attributes.
        self.temp_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "SHOE_RED",
                "title": "Red running shoe",
                "categories": ["Clothing", "Shoes"],
                "features": ["lightweight", "cotton"],
                "details": {"color": "red"},
                "store": "Example Store",
                "description": ["shoe for daily running"],
            },
            {
                "parent_asin": "BOOT_BLUE",
                "title": "Blue winter boot",
                "categories": ["Clothing", "Boots"],
                "features": ["warm", "leather"],
                "details": {"color": "blue"},
                "store": "Example Store",
                "description": ["boot for cold weather"],
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)

    def tearDown(self) -> None:
        # Close the database and remove the temporary catalog after each test.
        self.agent.connection.close()
        self.temp_directory.cleanup()

    def test_sessions_keep_separate_conversation_history(self) -> None:
        # Messages from one customer should not appear in another session.
        self.agent.reset("customer_one", {})
        self.agent.reset("customer_two", {})

        self.agent.respond("customer_one", "I need running shoes.", 1, 10)

        self.assertEqual(self.agent._sessions["customer_two"]["messages"], [])
        self.assertIn("running shoes", self.agent._sessions["customer_one"]["messages"][0])

    def test_question_order_starts_with_high_information_attributes(self) -> None:
        # Check that the agent asks broader questions before narrower ones.
        self.agent.reset("customer", {})

        questions = [
            self.agent.respond("customer", "I am still deciding.", turn, 10)["ask_attribute"]
            for turn in range(1, 6)
        ]

        self.assertEqual(questions, ["feature", "material", "color", "style", "size"])

    def test_later_answers_refine_the_original_request(self) -> None:
        # The second answer should be combined with the original product type.
        self.agent.reset("customer", {})

        self.agent.respond("customer", "I am looking for shoes.", 1, 10)
        response = self.agent.respond("customer", "Cotton matters to me.", 2, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "SHOE_RED")

    def test_intent_override_removes_the_old_preference(self) -> None:
        # When the customer changes their mind, the old colour should be removed.
        self.agent.reset("customer", {})

        self.agent.respond("customer", "I'm looking for shoes. I prefer blue.", 1, 10)
        self.agent.respond(
            "customer",
            "Actually, ignore my earlier preference. What I need is: red.",
            2,
            10,
        )

        memory = " ".join(self.agent._sessions["customer"]["messages"]).lower()
        self.assertIn("shoes", memory)
        self.assertIn("red", memory)
        self.assertNotIn("blue", memory)

    def test_agent_recommends_while_asking_a_question(self) -> None:
        # A follow-up question should not stop the agent from recommending products.
        self.agent.reset("customer", {})

        response = self.agent.respond("customer", "I need a red running shoe.", 1, 10)

        self.assertEqual(response["ask_attribute"], "feature")
        self.assertGreater(len(response["recommendations"]), 0)


if __name__ == "__main__":
    unittest.main()
