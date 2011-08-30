This folder contains source and compiled modules for a Gephi plugin. Note that
Gephi and its graph streaming plugin must be installed for this plugin to work.

To install, open Gephi, then the 'Tools' drop-down menu, 'Plugins', the
'Downloaded' tab, 'Add Plugins...', and browse to this folder to select the
'org-tulip-automatonsimulation.nbm' file.

# NOTE: For now, also select the 'org-gephi-layout-plugin.nbm' file. It
# contains an updated layout module that's necessary for the AutomatonSimulation
# plugin to work. (The current Gephi build uses version 0.8.0.2; this version
# is 0.8.0.3.)
#
# This extra step should become unnecessary once Gephi updates its release.

Then choose 'Install' and follow the instructions given.

---

To modify the source and recompile, you will need to download a development
branch of Gephi (specifically, Andre' Panisson's Graph Streaming branch)
and open the project in NetBeans. Follow these directions to get started:

http://wiki.gephi.org/index.php/Plugins_portal

The branch you need will be 'lp:~panisson/gephi/graphstreaming' instead of
'lp:gephi'.
