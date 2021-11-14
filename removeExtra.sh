LANGS="en_lines el_gdt es_gsd fr_gsd tr_imst mr_ufal hi_hdtb yo_ytb_auto br_keb_auto bxr_bdt_auto gsw_uzh_auto tl_trg_auto th_pud_auto cy_ccg_auto fo_oft_auto pom_bg_auto"
for lang in $LANGS
do
	rm -rf */Agreement/Gender/*freq */Agreement/Gender/*feats
	rm -rf */Agreement/Person/*freq */Agreement/Person/*feats
	rm -rf */Agreement/Number/*freq */Agreement/Number/*feats
	rm -rf */Agreement/Tense/*freq */Agreement/Tense/*feats
	rm -rf */Agreement/Mood/*freq */Agreement/Mood/*feats
	rm -rf */Agreement/*txt
	rm -rf */Agreement/*feats

	rm -rf */WordOrder/*freq */WordOrder/*feats */WordOrder/*txt
	rm -rf */WordOrder/*/*freq */WordOrder/*/*feats* */WordOrder/*/*txt*
	
	rm -rf */CaseMarking/*freq */CaseMarking/*feats */CaseMarking/*txt
	rm -rf */CaseMarking/*/*freq */CaseMarking/*/*feats */CaseMarking/*/*txt

	rm -rf */*txt
	rm -rf */index_agreement.html */index_wordorder.html
done

