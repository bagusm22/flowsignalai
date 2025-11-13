#!/usr/bin/env python3
"""
Hybrid Breakout Bullish Prediction v4
- Silent yfinance + clean universe
- Follow-through labeling (lebih ketat, kurangi false breakout)
- XGBoost dengan scale_pos_weight
- Skor hybrid TANPA squeeze (lebih cepat tangkap breakout)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import contextlib
import logging
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# CONFIG
# =========================
UNIVERSE = [
    "AALI.JK",
    "ABBA.JK",
    "ABDA.JK",
    "ABM.JK",
    "ABMM.JK",
    "ACES.JK",
    "ACRO.JK",
    "ACST.JK",
    "ADCP.JK",
    "ADES.JK",
    "ADHI.JK",
    "ADMF.JK", 
    "ADMG.JK",
    "ADMR.JK",
    "ADRO.JK",
    "AEGS.JK",
    "AGAR.JK",
    "AGII.JK",
    "AGRO.JK",
    "AGRS.JK",
    "AHAP.JK",
    "AIDX.JK",
    "AIMS.JK",
    "AISA.JK", 
    "AKKU.JK",
    "AKPI.JK",
    "AKR.JK",
    "AKRA.JK",
    "AKSI.JK",
    "ALDO.JK",
    "ALKA.JK",
    "ALMI.JK",
    "ALTO.JK",
    "AMAG.JK",
    "AMAN.JK",
    "AMAR.JK", 
    "AMFG.JK",
    "AMIN.JK",
    "AMMN.JK",
    "AMMS.JK",
    "AMOR.JK",
    "AMRT.JK",
    "ANDI.JK",
    "ANJT.JK",
    "ANTM.JK",
    "APEX.JK",
    "APIC.JK",
    "APII.JK", 
    "APLI.JK",
    "APLN.JK",
    "APSI.JK",
    "ARCI.JK",
    "AREA.JK",
    "ARGO.JK",
    "ARII.JK",
    "ARKA.JK",
    "ARKO.JK",
    "ARMY.JK",
    "ARNA.JK",
    "ARTA.JK", 
    "ARTI.JK",
    "ARTO.JK",
    "ASBI.JK",
    "ASDM.JK",
    "ASE.JK",
    "ASGR.JK",
    "ASHA.JK",
    "ASIA.JK",
    "ASII.JK",
    "ASJT.JK",
    "ASLC.JK",
    "ASLI.JK", 
    "ASMI.JK",
    "ASPI.JK",
    "ASRI.JK",
    "ASRM.JK",
    "ASSA.JK",
    "ATAP.JK",
    "ATIC.JK",
    "ATLA.JK",
    "AVIA.JK",
    "AWAN.JK",
    "AXIO.JK",
    "AYAM.JK", 
    "AYLS.JK",
    "BABP.JK",
    "BABY.JK",
    "BACA.JK",
    "BAIK.JK",
    "BAJA.JK",
    "BALI.JK",
    "BAPA.JK",
    "BAPI.JK",
    "BATA.JK",
    "BATR.JK",
    "BAUT.JK", 
    "BAYU.JK",
    "BBCA.JK",
    "BBHI.JK",
    "BBKP.JK",
    "BBLD.JK",
    "BBMD.JK",
    "BBNI.JK",
    "BBRI.JK",
    "BBRM.JK",
    "BBSI.JK",
    "BBSS.JK",
    "BBTN.JK", 
    "BBYB.JK",
    "BCAP.JK",
    "BCIC.JK",
    "BCIP.JK",
    "BDKR.JK",
    "BDMN.JK",
    "BEBS.JK",
    "BEEF.JK",
    "BEI.JK",
    "BEKS.JK",
    "BELI.JK",
    "BELL.JK", 
    "BESS.JK",
    "BEST.JK",
    "BFI.JK",
    "BFIN.JK",
    "BGTG.JK",
    "BHAT.JK",
    "BHIT.JK",
    "BIKA.JK",
    "BIKE.JK",
    "BIMA.JK",
    "BINA.JK",
    "BINO.JK", 
    "BIPI.JK",
    "BIPP.JK",
    "BIRD.JK",
    "BISI.JK",
    "BIST.JK",
    "BJBR.JK",
    "BJTM.JK",
    "BKDP.JK",
    "BKSL.JK",
    "BKSW.JK",
    "BLES.JK",
    "BLTA.JK", 
    "BLTZ.JK",
    "BLUE.JK",
    "BMAS.JK",
    "BMBL.JK",
    "BMHS.JK",
    "BMRI.JK",
    "BMSR.JK",
    "BMTR.JK",
    "BNBA.JK",
    "BNBR.JK",
    "BNGA.JK",
    "BNII.JK", 
    "BNLI.JK",
    "BOBA.JK",
    "BOGA.JK",
    "BOLA.JK",
    "BOLT.JK",
    "BOSS.JK",
    "BPFI.JK",
    "BPII.JK",
    "BPTR.JK",
    "BRAM.JK",
    "BRIS.JK",
    "BRMS.JK", 
    "BRNA.JK",
    "BRPT.JK",
    "BSBK.JK",
    "BSDE.JK",
    "BSE.JK",
    "BSIM.JK",
    "BSML.JK",
    "BSSR.JK",
    "BSWD.JK",
    "BTEK.JK",
    "BTEL.JK",
    "BTON.JK", 
    "BTPN.JK",
    "BTPS.JK",
    "BUAH.JK",
    "BUDI.JK",
    "BUKA.JK",
    "BUKK.JK",
    "BULL.JK",
    "BUMI.JK",
    "BUVA.JK",
    "BVIC.JK",
    "BWPT.JK",
    "BYAN.JK", 
    "CAKK.JK",
    "CAMP.JK",
    "CANI.JK",
    "CARE.JK",
    "CARS.JK",
    "CASA.JK",
    "CASH.JK",
    "CASS.JK",
    "CBMF.JK",
    "CBP.JK",
    "CBPE.JK",
    "CBRE.JK", 
    "CBUT.JK",
    "CCSI.JK",
    "CEKA.JK",
    "CENT.JK",
    "CFIN.JK",
    "CGAS.JK",
    "CHEM.JK",
    "CHIP.JK",
    "CIMB.JK",
    "CINT.JK",
    "CITA.JK",
    "CITY.JK", 
    "CLAY.JK",
    "CLEO.JK",
    "CLPI.JK",
    "CMNP.JK",
    "CMNT.JK",
    "CMPP.JK",
    "CMRY.JK",
    "CNBC.JK",
    "CNKO.JK",
    "CNMA.JK",
    "CNN.JK",
    "CNTX.JK", 
    "COAL.JK",
    "COCO.JK",
    "COWL.JK",
    "CPIN.JK",
    "CPRI.JK",
    "CPRO.JK",
    "CPU.JK",
    "CRAB.JK",
    "CRSN.JK",
    "CSAP.JK",
    "CSI.JK",
    "CSIS.JK", 
    "CSMI.JK",
    "CSRA.JK",
    "CTBN.JK",
    "CTRA.JK",
    "CTTH.JK",
    "CUAN.JK",
    "CYBR.JK",
    "DADA.JK",
    "DART.JK",
    "DATA.JK",
    "DAYA.JK",
    "DCI.JK", 
    "DCII.JK",
    "DEAL.JK",
    "DEFI.JK",
    "DEPO.JK",
    "DEWA.JK",
    "DEWI.JK",
    "DFAM.JK",
    "DFM.JK",
    "DGIK.JK",
    "DGNS.JK",
    "DIGI.JK",
    "DILD.JK", 
    "DIVA.JK",
    "DKFT.JK",
    "DLTA.JK",
    "DMAS.JK",
    "DMMX.JK",
    "DMND.JK",
    "DMS.JK",
    "DNAR.JK",
    "DNET.JK",
    "DOID.JK",
    "DOOH.JK",
    "DOSS.JK", 
    "DPNS.JK",
    "DPUM.JK",
    "DRMA.JK",
    "DSE.JK",
    "DSFI.JK",
    "DSNG.JK",
    "DSSA.JK",
    "DUCK.JK",
    "DUTI.JK",
    "DVLA.JK",
    "DWGL.JK",
    "DYAN.JK", 
    "EAST.JK",
    "ECII.JK",
    "EDGE.JK",
    "EJKSE.JK",
    "EKAD.JK",
    "ELIT.JK",
    "ELPI.JK",
    "ELSA.JK",
    "ELTY.JK",
    "EMDE.JK",
    "EMTK.JK",
    "ENAK.JK", 
    "ENRG.JK",
    "ENVY.JK",
    "ENZO.JK",
    "EPAC.JK",
    "EPMT.JK",
    "ERAA.JK",
    "ERAL.JK",
    "ERTX.JK",
    "ESIP.JK",
    "ESSA.JK",
    "ESTA.JK",
    "ESTI.JK", 
    "ETWA.JK",
    "EURO.JK",
    "EXCL.JK",
    "FAP.JK",
    "FAPA.JK",
    "FAST.JK",
    "FASW.JK",
    "FIDX.JK",
    "FILM.JK",
    "FIMP.JK",
    "FIRE.JK",
    "FISH.JK", 
    "FITT.JK",
    "FKS.JK",
    "FKX.JK",
    "FLMC.JK",
    "FMII.JK",
    "FOLK.JK",
    "FORU.JK",
    "FORZ.JK",
    "FPNI.JK",
    "FREN.JK",
    "FUJI.JK",
    "FUTR.JK", 
    "FWCT.JK",
    "GAMA.JK",
    "GDST.JK",
    "GDYR.JK",
    "GEH.JK",
    "GEL.JK",
    "GEMA.JK",
    "GEMS.JK",
    "GES.JK",
    "GET.JK",
    "GGRM.JK",
    "GGRP.JK", 
    "GHON.JK",
    "GIAA.JK",
    "GIFT.JK",
    "GJTL.JK",
    "GLOB.JK",
    "GLVA.JK",
    "GMFI.JK",
    "GMTD.JK",
    "GOLD.JK",
    "GOLF.JK",
    "GOLL.JK",
    "GOOD.JK", 
    "GOTO.JK",
    "GPRA.JK",
    "GPSO.JK",
    "GRIA.JK",
    "GRPH.JK",
    "GRPM.JK",
    "GSMF.JK",
    "GTBO.JK",
    "GTS.JK",
    "GTSI.JK",
    "GULA.JK",
    "GUNA.JK", 
    "GWSA.JK",
    "GZCO.JK",
    "HADE.JK",
    "HAIS.JK",
    "HAJJ.JK",
    "HALO.JK",
    "HATM.JK",
    "HBAT.JK",
    "HDFA.JK",
    "HDIT.JK",
    "HDTX.JK",
    "HEAL.JK", 
    "HELI.JK",
    "HERO.JK",
    "HEXA.JK",
    "HITS.JK",
    "HKMU.JK",
    "HMSP.JK",
    "HOKI.JK",
    "HOME.JK",
    "HOMI.JK",
    "HOPE.JK",
    "HOTL.JK",
    "HRME.JK", 
    "HRTA.JK",
    "HRUM.JK",
    "HUMI.JK",
    "HYGN.JK",
    "IATA.JK",
    "IBFN.JK",
    "IBK.JK",
    "IBOS.JK",
    "IBST.JK",
    "ICBP.JK",
    "ICON.JK",
    "ICTSI.JK", 
    "IDEA.JK",
    "IDPR.JK",
    "IFII.JK",
    "IFSH.JK",
    "IGAR.JK",
    "IIKP.JK",
    "IKAI.JK",
    "IKAN.JK",
    "IKBI.JK",
    "IKPM.JK",
    "IMAS.JK",
    "IMC.JK", 
    "IMJS.JK",
    "IMPC.JK",
    "INAF.JK",
    "INAI.JK",
    "INCF.JK",
    "INCI.JK",
    "INCO.JK",
    "IND.JK",
    "INDEX.JK",
    "INDF.JK",
    "INDR.JK",
    "INDS.JK", 
    "INDX.JK",
    "INDY.JK",
    "INET.JK",
    "INKP.JK",
    "INOV.JK",
    "INPC.JK",
    "INPP.JK",
    "INPS.JK",
    "INRU.JK",
    "INTA.JK",
    "INTD.JK",
    "INTP.JK", 
    "IOTF.JK",
    "IPAC.JK",
    "IPCC.JK",
    "IPCM.JK",
    "IPOL.JK",
    "IPPE.JK",
    "IPTV.JK",
    "IRRA.JK",
    "IRSX.JK",
    "ISAP.JK",
    "ISAT.JK",
    "ISBN.JK", 
    "ISSI.JK",
    "ISSP.JK",
    "ITIC.JK",
    "ITMA.JK",
    "ITMG.JK",
    "ITSEC.JK",
    "JARR.JK",
    "JAST.JK",
    "JATI.JK",
    "JAWA.JK",
    "JECC.JK",
    "JGLE.JK", 
    "JIHD.JK",
    "JKON.JK",
    "JKSW.JK",
    "JMAS.JK",
    "JPFA.JK",
    "JRPT.JK",
    "JSKY.JK",
    "JSMR.JK",
    "JSPT.JK",
    "JSX.JK",
    "JTPE.JK",
    "KAEF.JK", 
    "KARW.JK",
    "KASE.JK",
    "KAYU.JK",
    "KBAG.JK",
    "KBLI.JK",
    "KBLM.JK",
    "KBLV.JK",
    "KBRI.JK",
    "KDB.JK",
    "KDSI.JK",
    "KDTN.JK",
    "KEEN.JK", 
    "KEJU.JK",
    "KETR.JK",
    "KIAS.JK",
    "KICI.JK",
    "KIJA.JK",
    "KING.JK",
    "KINO.JK",
    "KIOS.JK",
    "KJEN.JK",
    "KKES.JK",
    "KKGI.JK",
    "KLAS.JK", 
    "KLBF.JK",
    "KLCI.JK",
    "KLIN.JK",
    "KMDS.JK",
    "KMI.JK",
    "KMTR.JK",
    "KOBX.JK",
    "KOCI.JK",
    "KOIN.JK",
    "KOKA.JK",
    "KONI.JK",
    "KOPI.JK", 
    "KOSPI.JK",
    "KPAL.JK",
    "KPAS.JK",
    "KPIG.JK",
    "KRAH.JK",
    "KRAS.JK",
    "KREN.JK",
    "KRYA.JK",
    "KSE.JK",
    "KUAS.JK",
    "LABA.JK",
    "LABS.JK", 
    "LAJU.JK",
    "LAND.JK",
    "LAPD.JK",
    "LCGP.JK",
    "LCK.JK",
    "LCKM.JK",
    "LEAD.JK",
    "LFLO.JK",
    "LIFE.JK",
    "LINK.JK",
    "LION.JK",
    "LIVE.JK", 
    "LMAS.JK",
    "LMAX.JK",
    "LMPI.JK",
    "LMSH.JK",
    "LOPI.JK",
    "LPCK.JK",
    "LPGI.JK",
    "LPIN.JK",
    "LPKR.JK",
    "LPLI.JK",
    "LPPF.JK",
    "LPPS.JK", 
    "LRNA.JK",
    "LSIP.JK",
    "LTLS.JK",
    "LUCK.JK",
    "LUCY.JK",
    "MABA.JK",
    "MAGP.JK",
    "MAHA.JK",
    "MAIN.JK",
    "MAMI.JK",
    "MANG.JK",
    "MAP.JK", 
    "MAPA.JK",
    "MAPB.JK",
    "MAPI.JK",
    "MARI.JK",
    "MARK.JK",
    "MASA.JK",
    "MASB.JK",
    "MAXI.JK",
    "MAYA.JK",
    "MBAP.JK",
    "MBMA.JK",
    "MBSS.JK", 
    "MBTO.JK",
    "MCAS.JK",
    "MCOL.JK",
    "MCOR.JK",
    "MDIA.JK",
    "MDKA.JK",
    "MDKI.JK",
    "MDLN.JK",
    "MDRN.JK",
    "MEDC.JK",
    "MEDS.JK",
    "MEGA.JK", 
    "MEJA.JK",
    "MENN.JK",
    "MERK.JK",
    "META.JK",
    "MFD.JK",
    "MFIN.JK",
    "MFMI.JK",
    "MGLV.JK",
    "MGNA.JK",
    "MGRO.JK",
    "MHKI.JK",
    "MICE.JK", 
    "MIDI.JK",
    "MIKA.JK",
    "MINA.JK",
    "MIRA.JK",
    "MITI.JK",
    "MKAP.JK",
    "MKNT.JK",
    "MKPI.JK",
    "MKTR.JK",
    "MLBI.JK",
    "MLIA.JK",
    "MLPL.JK", 
    "MLPT.JK",
    "MMIX.JK",
    "MMLP.JK",
    "MNC.JK",
    "MNCN.JK",
    "MOLI.JK",
    "MORA.JK",
    "MPIX.JK",
    "MPMX.JK",
    "MPOW.JK",
    "MPPA.JK",
    "MPRO.JK", 
    "MPX.JK",
    "MPXL.JK",
    "MRAT.JK",
    "MREI.JK",
    "MSIE.JK",
    "MSIG.JK",
    "MSIN.JK",
    "MSJA.JK",
    "MSKY.JK",
    "MSTI.JK",
    "MTDL.JK",
    "MTEL.JK", 
    "MTFN.JK",
    "MTLA.JK",
    "MTMH.JK",
    "MTPS.JK",
    "MTRA.JK",
    "MTSM.JK",
    "MTWI.JK",
    "MUTU.JK",
    "MYOH.JK",
    "MYOR.JK",
    "MYRX.JK",
    "MYTX.JK", 
    "NANO.JK",
    "NASA.JK",
    "NASI.JK",
    "NATO.JK",
    "NAYZ.JK",
    "NELY.JK",
    "NEST.JK",
    "NETV.JK",
    "NFC.JK",
    "NFCX.JK",
    "NICE.JK",
    "NICK.JK", 
    "NICL.JK",
    "NIFTY.JK",
    "NIKL.JK",
    "NINE.JK",
    "NIPS.JK",
    "NIRO.JK",
    "NISP.JK",
    "NOBU.JK",
    "NPGF.JK",
    "NRCA.JK",
    "NTBK.JK",
    "NUSA.JK", 
    "NZIA.JK",
    "OASA.JK",
    "OBM.JK",
    "OBMD.JK",
    "OCAP.JK",
    "OCBC.JK",
    "OILS.JK",
    "OKAS.JK",
    "OLIV.JK",
    "OMED.JK",
    "OMRE.JK",
    "OPMS.JK", 
    "PACK.JK",
    "PADA.JK",
    "PADI.JK",
    "PALM.JK",
    "PAM.JK",
    "PAMG.JK",
    "PANI.JK",
    "PANR.JK",
    "PANS.JK",
    "PBID.JK",
    "PBRX.JK",
    "PBSA.JK", 
    "PCAR.JK",
    "PDES.JK",
    "PDPP.JK",
    "PEGE.JK",
    "PEHA.JK",
    "PEVE.JK",
    "PGAS.JK",
    "PGEO.JK",
    "PGJO.JK",
    "PGLI.JK",
    "PGUN.JK",
    "PICO.JK", 
    "PIPA.JK",
    "PJAA.JK",
    "PKPK.JK",
    "PLAN.JK",
    "PLAS.JK",
    "PLIN.JK",
    "PMJS.JK",
    "PMMP.JK",
    "PNBN.JK",
    "PNBS.JK",
    "PNGO.JK",
    "PNIN.JK", 
    "PNLF.JK",
    "PNSE.JK",
    "POLA.JK",
    "POLI.JK",
    "POLL.JK",
    "POLU.JK",
    "POLY.JK",
    "POOL.JK",
    "PORT.JK",
    "POSA.JK",
    "POWR.JK",
    "PPGL.JK", 
    "PPRE.JK",
    "PPRI.JK",
    "PPRO.JK",
    "PRAS.JK",
    "PRAY.JK",
    "PRDA.JK",
    "PRIM.JK",
    "PSAB.JK",
    "PSDN.JK",
    "PSE.JK",
    "PSGO.JK",
    "PSKT.JK", 
    "PSSI.JK",
    "PTBA.JK",
    "PTDU.JK",
    "PTIS.JK",
    "PTMP.JK",
    "PTMR.JK",
    "PTPP.JK",
    "PTPS.JK",
    "PTPW.JK",
    "PTRO.JK",
    "PTSN.JK",
    "PTSP.JK", 
    "PUDP.JK",
    "PURA.JK",
    "PURE.JK",
    "PURI.JK",
    "PWON.JK",
    "PYFA.JK",
    "PZZA.JK",
    "QNB.JK",
    "RAAM.JK",
    "RAFI.JK",
    "RAJA.JK",
    "RALS.JK", 
    "RANC.JK",
    "RBMS.JK",
    "RCCC.JK",
    "RCR.JK",
    "RDTX.JK",
    "REAL.JK",
    "RELF.JK",
    "RELI.JK",
    "RGAS.JK",
    "RICY.JK",
    "RIGS.JK",
    "RIMO.JK", 
    "RISE.JK",
    "RLQ.JK",
    "RMBA.JK",
    "RMK.JK",
    "RMKE.JK",
    "ROCK.JK",
    "RODA.JK",
    "RONY.JK",
    "ROTI.JK",
    "RSCH.JK",
    "RSGK.JK",
    "RUIS.JK", 
    "RUNS.JK",
    "SAFE.JK",
    "SAGE.JK",
    "SAME.JK",
    "SAMF.JK",
    "SAPX.JK",
    "SATU.JK",
    "SBAT.JK",
    "SBMA.JK",
    "SCCO.JK",
    "SCMA.JK",
    "SCNP.JK", 
    "SCPI.JK",
    "SDMU.JK",
    "SDPC.JK",
    "SDRA.JK",
    "SEMA.JK",
    "SET.JK",
    "SFAN.JK",
    "SGER.JK",
    "SGRO.JK",
    "SHID.JK",
    "SHIP.JK",
    "SICO.JK", 
    "SIDO.JK",
    "SILO.JK",
    "SIMA.JK",
    "SIMP.JK",
    "SINI.JK",
    "SIPD.JK",
    "SKBM.JK",
    "SKLT.JK",
    "SKRN.JK",
    "SKYB.JK",
    "SLIS.JK",
    "SLJ.JK", 
    "SMAR.JK",
    "SMBR.JK",
    "SMCB.JK",
    "SMDM.JK",
    "SMDR.JK",
    "SMGA.JK",
    "SMGR.JK",
    "SMIL.JK",
    "SMKL.JK",
    "SMKM.JK",
    "SMLE.JK",
    "SMMA.JK", 
    "SMMT.JK",
    "SMR.JK",
    "SMRA.JK",
    "SMRU.JK",
    "SMSM.JK",
    "SNLK.JK",
    "SOCI.JK",
    "SOFA.JK",
    "SOHO.JK",
    "SOLA.JK",
    "SONA.JK",
    "SOSS.JK", 
    "SOTS.JK",
    "SOUL.JK",
    "SPMA.JK",
    "SPRE.JK",
    "SPTO.JK",
    "SQMI.JK",
    "SRAJ.JK",
    "SRIL.JK",
    "SRSN.JK",
    "SRTG.JK",
    "SSE.JK",
    "SSIA.JK", 
    "SSMS.JK",
    "SSTM.JK",
    "STAA.JK",
    "STAR.JK",
    "STI.JK",
    "STTP.JK",
    "SUGI.JK",
    "SULI.JK",
    "SUNI.JK",
    "SUPR.JK",
    "SURE.JK",
    "SURI.JK", 
    "SWAT.JK",
    "SWID.JK",
    "SZSE.JK",
    "TAIEX.JK",
    "TALF.JK",
    "TAMU.JK",
    "TAPG.JK",
    "TARA.JK",
    "TASI.JK",
    "TAXI.JK",
    "TAYS.JK",
    "TBIG.JK", 
    "TBLA.JK",
    "TBMS.JK",
    "TBS.JK",
    "TCID.JK",
    "TCPI.JK",
    "TDPM.JK",
    "TEBE.JK",
    "TECH.JK",
    "TELE.JK",
    "TFAS.JK",
    "TFCO.JK",
    "TGKA.JK", 
    "TGRA.JK",
    "TGUK.JK",
    "TIFA.JK",
    "TINS.JK",
    "TIRA.JK",
    "TIRT.JK",
    "TKIM.JK",
    "TLDN.JK",
    "TLKM.JK",
    "TMAS.JK",
    "TMPO.JK",
    "TNCA.JK", 
    "TOBA.JK",
    "TOOL.JK",
    "TOPIX.JK",
    "TOPS.JK",
    "TOSK.JK",
    "TOTL.JK",
    "TOTO.JK",
    "TOWR.JK",
    "TOYS.JK",
    "TPIA.JK",
    "TPMA.JK",
    "TRAM.JK", 
    "TRGU.JK",
    "TRIL.JK",
    "TRIM.JK",
    "TRIN.JK",
    "TRIO.JK",
    "TRIS.JK",
    "TRJA.JK",
    "TRON.JK",
    "TRST.JK",
    "TRUE.JK",
    "TRUK.JK",
    "TRUS.JK", 
    "TSE.JK",
    "TSPC.JK",
    "TUGU.JK",
    "TURI.JK",
    "TYRE.JK",
    "UANG.JK",
    "UBC.JK",
    "UCID.JK",
    "UDNG.JK",
    "UFOE.JK",
    "ULS.JK",
    "ULSP.JK", 
    "ULTJ.JK",
    "UNIC.JK",
    "UNIQ.JK",
    "UNIT.JK",
    "UNSP.JK",
    "UNTR.JK",
    "UNVR.JK",
    "URBN.JK",
    "URI.JK",
    "URL.JK",
    "UTC.JK",
    "UTF.JK", 
    "UVCR.JK",
    "VAST.JK",
    "VERN.JK",
    "VICI.JK",
    "VICO.JK",
    "VINS.JK",
    "VISI.JK",
    "VIVA.JK",
    "VKTR.JK",
    "VOKS.JK",
    "VRNA.JK",
    "WAPO.JK", 
    "WEGE.JK",
    "WEHA.JK",
    "WGSH.JK",
    "WICO.JK",
    "WIDI.JK",
    "WIFI.JK",
    "WIIM.JK",
    "WIKA.JK",
    "WINR.JK",
    "WINS.JK",
    "WIR.JK",
    "WIRG.JK", 
    "WMEP.JK",
    "WMES.JK",
    "WMPP.JK",
    "WMUU.JK",
    "WOMF.JK",
    "WOOD.JK",
    "WOWS.JK",
    "WSBP.JK",
    "WSKT.JK",
    "WTON.JK",
    "YELO.JK",
    "YPAS.JK", 
    "YULE.JK",
    "ZATA.JK",
    "ZBRA.JK",
    "ZINC.JK",
    "ZONE.JK",
    "ZYRX.JK",
]

PERIOD_DAILY     = "3y"
PERIOD_INTRADAY  = "30d"
INTERVAL_INTRADAY= "60m"

# nilai transaksi rata2 10 hari minimal (lembar * harga) → Rp 20 M
LIQ_THRESHOLD    = 2e10

# bobot skor hybrid (tanpa squeeze)
W_ML   = 0.49
W_VOL  = 0.33
W_TF   = 0.18

# =========================
# UTILS: silent yfinance
# =========================
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def safe_yf_download(*args, **kwargs):
    """Redir stdout/stderr agar '1 Failed download' tidak muncul"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            df = yf.download(*args, **kwargs)
        except Exception:
            return None
    if df is None or df.empty:
        return None
    return df

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def get_col(df: pd.DataFrame, name: str) -> pd.Series:
    if (df is None) or (name not in df.columns):
        return pd.Series(index=df.index, dtype=float)
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col

# =========================
# CLEAN UNIVERSE
# =========================
def clean_universe(universe, period="30d", interval="1d"):
    valid = []
    for sym in universe:
        df = safe_yf_download(
            tickers=sym,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df is not None and not df.empty:
            valid.append(sym)
    return valid

# =========================
# DOWNLOADERS
# =========================
def download_daily(symbol: str, period: str = PERIOD_DAILY):
    df = safe_yf_download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None:
        return None
    df = normalize_df(df)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Khusus .JK: volume di yfinance biasanya per LOT → ubah ke lembar
    if symbol.endswith(".JK") and "Volume" in df.columns:
        df["Volume"] = df["Volume"] * 100

    return df.dropna(how="any")

def download_intraday(symbol: str, period: str = PERIOD_INTRADAY, interval: str = INTERVAL_INTRADAY):
    df = safe_yf_download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None:
        return None
    df = normalize_df(df)
    if df.index.tz is not None:
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    return df.dropna(how="any")

# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, period: int = 20): return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, std_factor: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return upper, lower, ma

def atr(df: pd.DataFrame, period: int = 14):
    high = get_col(df, "High")
    low = get_col(df, "Low")
    close = get_col(df, "Close")
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def obv(df: pd.DataFrame):
    close = get_col(df, "Close")
    volume = get_col(df, "Volume").fillna(0)
    obv_vals = [0]
    for i in range(1, len(df)):
        c_now = close.iloc[i]; c_prev = close.iloc[i - 1]; v_now = volume.iloc[i]
        if c_now > c_prev: obv_vals.append(obv_vals[-1] + v_now)
        elif c_now < c_prev: obv_vals.append(obv_vals[-1] - v_now)
        else: obv_vals.append(obv_vals[-1])
    return pd.Series(obv_vals, index=df.index)

# =========================
# FEATURES PER SAHAM
# =========================
def build_features_for_symbol(symbol: str):
    # Daily
    daily = download_daily(symbol)
    if daily is None or daily.empty or len(daily) < 60:
        return None

    open_  = get_col(daily, "Open")
    high   = get_col(daily, "High")
    low    = get_col(daily, "Low")
    close  = get_col(daily, "Close")
    volume = get_col(daily, "Volume")

    # Teknis
    daily["EMA5"]  = ema(close, 5)
    daily["EMA20"] = ema(close, 20)
    daily["EMA50"] = ema(close, 50)
    daily["RSI14"] = rsi(close, 14)
    daily["ATR14"] = atr(daily, 14)

    upper, lower, mid = bollinger_bands(close, 20, 2.0)
    daily["BB_upper"] = upper
    daily["BB_lower"] = lower
    daily["BB_mid"]   = mid
    daily["BB_width"] = (upper - lower) / close

    daily["OBV"]       = obv(daily)
    daily["OBV_slope"] = daily["OBV"].diff()

    daily["vol_avg_10"]   = volume.rolling(10).mean()
    daily["volume_ratio"] = volume / daily["vol_avg_10"]
    daily["ATR_ratio"]    = daily["ATR14"] / close

    # Likuiditas: nilai transaksi rata2 10 hari
    daily["value_traded"] = close * volume
    daily["liq_score"]    = daily["value_traded"].rolling(10).mean()
    daily["is_liquid"]    = (daily["liq_score"] > LIQ_THRESHOLD).astype(int)

    # Candle shape
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    daily["body_ratio"] = (body / rng).clip(upper=1.0).fillna(0.0)
    daily["close_pos"]  = ((close - low) / (high - low).replace(0, np.nan)).clip(0,1).fillna(0.0)

    # Intraday TF confirm
    intra = download_intraday(symbol)
    if intra is not None and not intra.empty:
        close_h1 = get_col(intra, "Close")
        intra["EMA_H1_20"] = ema(close_h1, 20)
        intra_daily = intra.resample("1D").last()[["Close", "EMA_H1_20"]]
        intra_daily = intra_daily.rename(columns={"Close": "Close_H1"})
        daily = daily.join(intra_daily, how="left")
        daily["TF_confirm"] = ((daily["Close_H1"] > daily["EMA_H1_20"]) & (daily["Close"] > daily["EMA20"])).fillna(False).astype(int)
    else:
        daily["TF_confirm"] = 0

    # Label: FOLLOW-THROUGH (2-hari) → lebih realistis
    daily["Close_next"]  = daily["Close"].shift(-1)
    daily["High_next"]   = daily["High"].shift(-1)
    daily["Close_next2"] = daily["Close"].shift(-2)
    # breakout valid jika besok tutup > high hari ini DAN lusa masih tutup di atas tutup besok
    daily["label_breakout"] = ((daily["Close_next"] > daily["High"]) & (daily["Close_next2"] > daily["Close_next"])).astype(int)

    daily["symbol"] = symbol
    return daily

# =========================
# BUILD DATASET
# =========================
def build_dataset(universe):
    dfs = []
    for sym in universe:
        try:
            df_sym = build_features_for_symbol(sym)
            if df_sym is not None and not df_sym.empty:
                dfs.append(df_sym)
        except Exception:
            continue
    if not dfs:
        return None
    all_df = pd.concat(dfs, axis=0)

    need_cols = [
        "volume_ratio","BB_width","RSI14","ATR_ratio",
        "OBV_slope","body_ratio","close_pos","TF_confirm",
        "label_breakout","is_liquid","Close","High","Open","Low","Volume"
    ]
    all_df = all_df.dropna(subset=need_cols, how="any")
    return all_df

# =========================
# TRAIN MODEL
# =========================
def train_model(df: pd.DataFrame):
    feature_cols = [
        "volume_ratio",
        "BB_width",
        "RSI14",
        "OBV_slope",
        "ATR_ratio",
        "body_ratio",
        "close_pos",
        "TF_confirm",
    ]
    df_train = df.dropna(subset=feature_cols + ["label_breakout"])
    X = df_train[feature_cols].values
    y = df_train["label_breakout"].values

    # handle imbalance
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    spw = float(neg) / max(1.0, float(pos)) if pos > 0 else 1.0

    split = int(len(df_train) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=250,
            learning_rate=0.10,   # sedikit lebih agresif agar sebaran prob lebih lebar
            max_depth=5,          # sedikit lebih dalam
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            n_estimators_target=None,
        )
        model.fit(X_train, y_train)
        print(f"[INFO] Using XGBClassifier (scale_pos_weight={spw:.2f})")
    except Exception as e:
        print("[WARN] xgboost unavailable, fallback RandomForest:", e)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight={0:1.0, 1:spw}
        )
        model.fit(X_train, y_train)

    # Eval
    from sklearn.metrics import accuracy_score, f1_score
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        print(f"[EVAL] acc={acc:.3f} f1={f1:.3f} (test size={len(X_test)})")

    return model, feature_cols

# =========================
# HYBRID SCORE (tanpa squeeze)
# =========================
def hybrid_score(row: pd.Series, ml_prob: float):
    # sinyal real-time
    vol_spike  = 1 if float(row["volume_ratio"]) > 2.5 else 0
    tf_confirm = int(row["TF_confirm"])

    # base score (tanpa squeeze)
    score = (W_VOL * vol_spike) + (W_ML * ml_prob) + (W_TF * tf_confirm)

    # penalti candle lemah (body kecil → potensi distribusi)
    body_ratio = float(row.get("body_ratio", 0.0))
    if body_ratio < 0.35:
        score *= 0.85

    return score, vol_spike, tf_confirm

# =========================
# MAIN
# =========================
def main():
    print("[STEP] Cleaning universe...")
    clean_list = clean_universe(UNIVERSE, period="30d", interval="1d")
    if not clean_list:
        print("No valid tickers. Exit.")
        return
    print(f"[INFO] Universe valid: {len(clean_list)} / {len(UNIVERSE)}")

    print("[STEP] Building dataset...")
    df_all = build_dataset(clean_list)
    if df_all is None or df_all.empty:
        print("No data. Exit.")
        return

    print("[STEP] Training model...")
    model, feature_cols = train_model(df_all)

    latest = (
        df_all.sort_index()
             .groupby("symbol")
             .tail(1)
             .reset_index(drop=True)
    )

    results = []
    for _, row in latest.iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            ml_prob = float(model.predict_proba(X_row)[0, 1])
        else:
            ml_prob = float(model.predict(X_row)[0])

        score, vol_spike, tf_confirm = hybrid_score(row, ml_prob)

        results.append({
            "symbol":       row["symbol"],
            "close":        float(row["Close"]),
            "ml_prob":      ml_prob,
            "volume_ratio": float(row["volume_ratio"]),
            "BB_width":     float(row["BB_width"]),
            "tf_confirm":   tf_confirm,
            "is_liquid":    int(row.get("is_liquid", 0)),
            "score":        float(score),
        })

    df_res = pd.DataFrame(results).sort_values("score", ascending=False)

    print("\n=== HYBRID BREAKOUT (no squeeze) ===")
    for _, r in df_res.iterrows():
        flag = "🔥" if r["score"] >= 0.7 else ("🟡" if r["score"] >= 0.5 else "⚪")
        print(
            f"{flag} {r['symbol']:8s} | score={r['score']:.2f} "
            f"| ml={r['ml_prob']:.2f} | volx={r['volume_ratio']:.2f} "
            f"| tf={r['tf_confirm']} | liq={'✅' if r['is_liquid'] else '❌'} "
            f"| bbW={r['BB_width']:.3f} | close={r['close']:.2f}"
        )

    print("""
------------------------------------------
LEGEND:
🔥  Strong Breakout Candidate (score ≥ 0.7)
🟡  Moderate Setup (0.5 ≤ score < 0.7)
⚪  Low Confidence (score < 0.5)

Skor = 0.49*ml_prob + 0.33*vol_spike + 0.18*tf_confirm,
+ penalti likuiditas, + penalti candle lemah.
Label = follow-through (T+1 close > today's high & T+2 close > T+1).
------------------------------------------
""")

if __name__ == "__main__":
    main()
