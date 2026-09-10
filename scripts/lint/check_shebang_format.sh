#!/usr/bin/env bash

FAIL=0

# -I skips binaries: a smudged LFS zipapp has a real python3 shebang over binary data.
if grep -I '^#!.*python' $@ | grep -v '#!/usr/bin/env python3$'; then
  echo -e "Invalid shebang! Must use '#!/usr/bin/env python3'\n"
  FAIL=1
fi

if grep -I '^#!.*bash' $@ | grep -v '#!/usr/bin/env bash$'; then
  echo -e "Invalid shebang! Must use '#!/usr/bin/env bash'"
  FAIL=1
fi

exit $FAIL
