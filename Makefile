
.PHONY: install-gpt-cli
install-gpt-cli:
	cargo install --path .


.PHONY: uninstall-gpt-cli
uninstall-gpt-cli:
	rm $(which gpt-cli)
